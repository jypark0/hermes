import logging

import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RootMeanSquaredError, RunningAverage
from ignite.utils import setup_logger
from torch import nn


class ObjectsSplitHeads(nn.Module):
    def __init__(
        self,
        backbone,
        post_process_dims,
        post_process_activation,
    ):
        # assert post_process_dims[0] == backbone.out_dim
        # assert post_process_dims[0] == 3 * backbone.out_dim

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

        # Take rho0 features and pass through projection layer
        rho0 = x[:, :, 0]

        for layer in self.post_process_layers:
            rho0 = layer(rho0)

        rho1 = x[:, :, 1:3]
        # Take mean over channels
        rho1 = rho1.mean(1)

        return F.log_softmax(rho0, dim=1), rho1


class ObjectsSplitHeadsEngine:
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
        self.loader_keys = loader_keys

        def train_step(engine, batch):
            if (engine.state.iteration - 1) % grad_accum_steps == 0:
                optimizer.zero_grad()
            model.train()

            x, y = prepare_batch(batch, device=device)
            occupancy_pred, ori_pred = model(x)

            nll_loss = loss_fn[0](occupancy_pred, y[:, 0, 0].long())
            rmse_loss = torch.sqrt(loss_fn[1](ori_pred, y[:, 1, 1:]))

            loss = nll_loss + 10 * rmse_loss

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            loss.backward()
            if engine.state.iteration % grad_accum_steps == 0:
                optimizer.step()

            return nll_loss, rmse_loss

        self.trainer = Engine(train_step)

        RunningAverage(output_transform=lambda x: x[0]).attach(
            self.trainer, "batch_nll"
        )
        RunningAverage(output_transform=lambda x: x[1]).attach(
            self.trainer, "batch_rmse"
        )
        ProgressBar(disable=disable_tqdm).attach(
            self.trainer, ["batch_nll", "batch_rmse"]
        )

        def eval_step(engine, batch):
            model.eval()
            with torch.no_grad():
                x, y = prepare_batch(batch, device=device)
                occupancy_pred, ori_pred = model(x)

                return (
                    occupancy_pred,
                    y[:, 0, 0].long(),
                    ori_pred,
                    y[:, 1, 1:],
                )

        self.evaluators = {}
        for k in self.loader_keys:
            self.evaluators[k] = Engine(eval_step)

            Accuracy(output_transform=lambda y: (y[0], y[1])).attach(
                self.evaluators[k], "accuracy"
            )
            Loss(loss_fn[0], output_transform=lambda y: (y[0], y[1])).attach(
                self.evaluators[k], "nll"
            )
            RootMeanSquaredError(output_transform=lambda y: (y[2], y[3])).attach(
                self.evaluators[k], "rmse"
            )

    def set_epoch_loggers(self, loaders_dict):
        # Setup logging level
        setup_logger(name="ignite", level=logging.WARNING)
        self.trainer.logger = setup_logger(name="trainer", level=logging.WARNING)
        for k, evaluator in self.evaluators.items():
            evaluator.logger = setup_logger(name=k, level=logging.WARNING)

        def inner_log(engine, evaluator, tag):
            evaluator.run(loaders_dict[tag])
            metrics = evaluator.state.metrics
            print(
                f"{tag.upper()} Results - Epoch: {engine.state.epoch} "
                f"Avg accuracy: {metrics['accuracy']:.5f}, Avg nll: {metrics['nll']:.5f}, Avg rmse: {metrics['rmse']:.5f}"
            )

        # Evaluate over loaders_dict
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_results(engine):
            for k in self.loader_keys:
                if loaders_dict[k] is not None:
                    inner_log(engine, self.evaluators[k], k)

    def create_wandb_logger(self, log_interval=1, optimizer=None, **kwargs):
        wandb_logger = WandBLogger(**kwargs)

        # Attach the logger to the trainer to log training loss at each iteration
        wandb_logger.attach_output_handler(
            self.trainer,
            event_name=Events.ITERATION_COMPLETED(every=log_interval),
            tag="train",
            output_transform=lambda loss: {"batch_nll": loss[0], "batch_rmse": loss[1]},
            state_attributes=["epoch"],
        )

        # Attach the logger to the optimizer parameters handler
        wandb_logger.attach_opt_params_handler(
            self.trainer,
            event_name=Events.ITERATION_STARTED(every=100),
            optimizer=optimizer,
        )

        # Attach logger to evaluator on test dataset
        for k in self.loader_keys:
            wandb_logger.attach_output_handler(
                self.evaluators[k],
                event_name=Events.EPOCH_COMPLETED,
                tag=k,
                metric_names=["accuracy", "nll", "rmse"],
                global_step_transform=lambda *_: self.trainer.state.iteration,
            )

        return wandb_logger
