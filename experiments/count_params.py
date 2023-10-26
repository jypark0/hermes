import hydra
from hydra.utils import instantiate

from src.run.utils import numel, set_seed


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg):
    set_seed(cfg.seed)

    # Create backbone and model
    backbone = instantiate(cfg.backbone.net).to(cfg.device)
    model = instantiate(cfg.model, backbone=backbone).to(cfg.device)

    num_params = numel(model, only_trainable=True)

    print(
        f"[{model.__class__.__name__}, Backbone={backbone.__class__.__name__}] No. trainable parameters = {num_params}"
    )


if __name__ == "__main__":
    main()
