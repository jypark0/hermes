import hydra
from hydra.utils import instantiate
from ignite.engine import Events
from ignite.handlers import LRScheduler, ModelCheckpoint
from omegaconf import OmegaConf

from src.run.utils import create_dataset_loaders, numel, prepare_batch_fn, set_seed


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg):
    set_seed(cfg.seed)

    loaders_dict = create_dataset_loaders(cfg)

    # Create backbone and model
    backbone = instantiate(cfg.backbone.net).to(cfg.device)
    model = instantiate(cfg.model, backbone=backbone).to(cfg.device)

    # FIXME Merge SpiralNet with backbone + model
    # if cfg.model.net._target_ == "src.model.spiralnet.SpiralNet":
    #     d = next(iter(loaders_dict["train"]))
    #     spiral_indices = preprocess_spiral(d.face.T, cfg.model.seq_length).to(
    #         cfg.device
    #     )
    #     model = instantiate(
    #         cfg.model.net, target_dim=cfg.dataset.num_nodes, indices=spiral_indices
    #     ).to(cfg.device)
    # else:
    #     model = instantiate(cfg.model.net).to(cfg.device)

    num_params = numel(model, only_trainable=True)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    if cfg.get("scheduler"):
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    else:
        scheduler = None

    loss_fn = instantiate(cfg.loss)
    prepare_batch = prepare_batch_fn(key="y")

    engine = instantiate(
        cfg.engine,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        prepare_batch=prepare_batch,
        loader_keys=loaders_dict.keys(),
    )

    engine.set_epoch_loggers(loaders_dict)
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_config["num_params"] = num_params
    _ = engine.create_wandb_logger(
        log_interval=1, optimizer=optimizer, config=wandb_config, **cfg.wandb
    )

    if cfg.get("save_dir"):
        gst = lambda *_: engine.trainer.state.epoch
        checkpoint_handler = ModelCheckpoint(
            cfg.save_dir,
            filename_prefix=f"{cfg.dataset.name}_{cfg.backbone.name}_seed{cfg.seed}",
            n_saved=1,
            require_empty=False,
            global_step_transform=gst,
            filename_pattern="{filename_prefix}_{name}.pt",
        )
        engine.trainer.add_event_handler(
            event_name=Events.EPOCH_COMPLETED,
            handler=checkpoint_handler,
            to_save={"model": model},
        )

    # Set LR scheduler if needed (for SpiralNet)
    if scheduler:
        ignite_scheduler = LRScheduler(scheduler)
        engine.trainer.add_event_handler(Events.EPOCH_STARTED, ignite_scheduler)

    print(
        f"[{model.__class__.__name__}, Backbone={backbone.__class__.__name__}] No. trainable parameters = {num_params}"
    )

    print(f"Transforms: {backbone.transforms}")
    print(f"Model: {model}")

    engine.trainer.run(loaders_dict["train"], max_epochs=cfg.train.epochs)


if __name__ == "__main__":
    main()
