import logging
from math import pi

import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import (
    Engine,
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import setup_logger
from torch import nn

from src.utils.rep_act import rep_act
from src.run.utils import GaugeInvarianceNLLLoss
from src.transform.gauge_transformer import GaugeTransformer


class FAUSTClassifier(nn.Module):
    def __init__(
        self,
        backbone,
        post_process_dims,
        post_process_activation,
    ):
        assert post_process_dims[0] == backbone.out_dim

        super().__init__()

        self.backbone = backbone

        self.post_process_layers = nn.ModuleList()
        for i in range(len(post_process_dims) - 2):
            self.post_process_layers.append(
                nn.Linear(
                    post_process_dims[i],
                    post_process_dims[i + 1],
                )
            )
            self.post_process_layers.append(post_process_activation)
            self.post_process_layers.append(nn.Dropout())
        self.post_process_layers.append(
            nn.Linear(
                post_process_dims[-2],
                post_process_dims[-1],
            )
        )

    def forward(self, x):
        x = self.backbone(x)

        # Take trivial feature
        # out_order of x = 0 anyway
        x = x[:, :, 0]

        for layer in self.post_process_layers:
            x = layer(x)

        return F.log_softmax(x, dim=1)


class FAUSTEngine:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        prepare_batch,
        loader_keys,
        disable_tqdm,
        grad_accum_steps=1,
    ):
        self.trainer = create_supervised_trainer(
            model,
            optimizer,
            loss_fn,
            device=device,
            prepare_batch=prepare_batch,
            gradient_accumulation_steps=grad_accum_steps,
        )
        self.loader_keys = loader_keys

        RunningAverage(output_transform=lambda x: x).attach(self.trainer, "loss")
        # GpuInfo().attach(self.trainer, name="gpu")
        # gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        # ProgressBar().attach(
        # 	  self.trainer,
        # 	  ["loss", f"gpu:{gpu_index} mem(%)", f"gpu:{gpu_index} util(%)"],
        # )
        ProgressBar(disable=disable_tqdm).attach(self.trainer, ["loss"])

        metrics_dict = {"nll": Loss(loss_fn), "accuracy": Accuracy()}
        self.evaluator = create_supervised_evaluator(
            model, metrics=metrics_dict, device=device, prepare_batch=prepare_batch
        )

        # FIXME doesn't work with RelTanTransform yet
        # Create evaluator for test_tf
        def eval_gauge_error(engine, batch):
            transform_angle = 2 * pi * torch.rand(batch.pos.shape[0]).to(device)
            transform = GaugeTransformer(transform_angle)

            model.eval()

            with torch.no_grad():
                x, y = batch.to(device), batch.y.to(device)
                y_orig = model(x)

                # Transformed inputs
                x_t = transform(x)
                x_t.x = rep_act(x_t.x, -transform_angle)
                y_t = model(x_t)

            return y_orig, y_t, y

        self.evaluator_gauge = Engine(eval_gauge_error)
        eval_tf_metric = GaugeInvarianceNLLLoss(device=device)
        eval_tf_metric.attach(self.evaluator_gauge, "gauge")

    def set_epoch_loggers(self, loaders_dict):
        # Setup logging level
        setup_logger(name="ignite", level=logging.WARNING)
        self.trainer.logger = setup_logger(name="trainer", level=logging.WARNING)
        self.evaluator.logger = setup_logger(name="evaluator", level=logging.WARNING)
        self.evaluator_gauge.logger = setup_logger(
            name="evaluator_gauge", level=logging.WARNING
        )

        def inner_log(engine, evaluator, tag):
            evaluator.run(loaders_dict[tag])
            metrics = evaluator.state.metrics
            if tag in ["train", "test"]:
                print(
                    f"{tag.upper()} Results - Epoch: {engine.state.epoch} "
                    f"Avg accuracy: {metrics['accuracy']:.5f} Avg loss: {metrics['nll']:.5f}"
                )
            elif tag == "test_gauge":
                print(
                    f"{tag.upper()} Results - Epoch: {engine.state.epoch} "
                    f"Avg gauge_invariance_nll_loss: {metrics['gauge']:.5f}"
                )

        # Evaluate on train dataset
        if loaders_dict["train"] is not None:

            @self.trainer.on(Events.EPOCH_COMPLETED)
            def log_train_results(engine):
                inner_log(engine, self.evaluator, "train")

        # Evaluate on test dataset
        if loaders_dict["test"] is not None:

            @self.trainer.on(Events.EPOCH_COMPLETED)
            def log_test_results(engine):
                inner_log(engine, self.evaluator, "test")

        # Evaluate on test dataset for GaugeInvarianceNLLLoss
        if loaders_dict["test_gauge"] is not None:

            @self.trainer.on(Events.EPOCH_COMPLETED)
            def log_test_gauge(engine):
                inner_log(engine, self.evaluator_gauge, "test_gauge")

    def create_wandb_logger(self, log_interval=1, optimizer=None, **kwargs):
        wandb_logger = WandBLogger(**kwargs)

        # Attach the logger to the trainer to log training loss at each iteration
        wandb_logger.attach_output_handler(
            self.trainer,
            event_name=Events.ITERATION_COMPLETED(every=log_interval),
            tag="train",
            output_transform=lambda loss: {"batch_nll": loss},
            state_attributes=["epoch"],
        )

        # Attach the logger to the optimizer parameters handler
        wandb_logger.attach_opt_params_handler(
            self.trainer,
            event_name=Events.ITERATION_STARTED(every=100),
            optimizer=optimizer,
        )

        for tag in ["train", "test"]:
            wandb_logger.attach_output_handler(
                self.evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["nll", "accuracy"],
                global_step_transform=lambda *_: self.trainer.state.iteration,
            )

        wandb_logger.attach_output_handler(
            self.evaluator_gauge,
            event_name=Events.EPOCH_COMPLETED,
            tag="test_gauge",
            metric_names=["gauge"],
            global_step_transform=lambda *_: self.trainer.state.iteration,
        )

        return wandb_logger
