import logging

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import RootMeanSquaredError, RunningAverage
from ignite.utils import setup_logger
from torch import nn


class PDERegressor(nn.Module):
    def __init__(
        self,
        backbone,
    ):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)

        # Take trivial feature
        x = x[:, :, 0]

        return x


class PDEEngine:
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
            # if engine.state.iteration > 100:
            #     return 0.0, 0.0
            if (engine.state.iteration - 1) % grad_accum_steps == 0:
                optimizer.zero_grad()
            model.train()

            x, yy = prepare_batch(batch, device=device)
            loss = 0
            for i in range(yy.shape[1]):
                y = yy[..., i].unsqueeze(-1)
                y_pred = model(x)
                x.x = torch.cat([x.x[:, y_pred.shape[1] :, 0], y_pred], 1)[:, :, None]
                loss += loss_fn(y_pred, y)
            loss /= yy.shape[1]

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            loss.backward()
            if engine.state.iteration % grad_accum_steps == 0:
                optimizer.step()

            return torch.sqrt(loss).item(), torch.linalg.norm(y_pred).item()

        self.trainer = Engine(train_step)

        RunningAverage(output_transform=lambda x: x[0]).attach(self.trainer, "loss")
        RunningAverage(output_transform=lambda x: x[1]).attach(
            self.trainer, "frob_norm"
        )
        ProgressBar(disable=disable_tqdm).attach(self.trainer, ["loss", "frob_norm"])

        def eval_step(engine, batch):
            # if engine.state.iteration > 10:
            #     return torch.zeros(1), -torch.ones(1)

            y_preds = []
            ys = []

            model.eval()
            with torch.no_grad():
                x, yy = prepare_batch(batch, device=device)
                for i in range(yy.shape[1]):
                    y = yy[..., i].unsqueeze(-1)
                    ys.append(y)
                    y_pred = model(x)
                    y_preds.append(y_pred)

                    x.x = torch.cat([x.x[:, y_pred.shape[1] :, 0], y_pred], 1)[
                        :, :, None
                    ]

                return torch.cat(y_preds, dim=-1), yy

        self.evaluators = {}
        for k in self.loader_keys:
            if k == "train":
                continue

            self.evaluators[k] = Engine(eval_step)

            metric = RootMeanSquaredError()
            metric.attach(self.evaluators[k], "rmse")

            RunningAverage(metric).attach(self.evaluators[k], "running_rmse")
            ProgressBar(persist=False, desc=k.upper(), disable=disable_tqdm).attach(
                self.evaluators[k], ["running_rmse"]
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
                f"Avg rmse: {metrics['rmse']:.5E}"
            )

        # Evaluate over loaders_dict
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_results(engine):
            for k in self.loader_keys:
                if k == "train":
                    continue
                if loaders_dict[k] is not None:
                    inner_log(engine, self.evaluators[k], k)

    def create_wandb_logger(self, log_interval=1, optimizer=None, **kwargs):
        wandb_logger = WandBLogger(**kwargs)

        # Attach the logger to the trainer to log training loss at each iteration
        wandb_logger.attach_output_handler(
            self.trainer,
            event_name=Events.ITERATION_COMPLETED(every=log_interval),
            tag="train",
            output_transform=lambda loss: {"batch_rmse": loss[0], "frob_norm": loss[1]},
            state_attributes=["epoch"],
        )

        # Attach the logger to the optimizer parameters handler
        wandb_logger.attach_opt_params_handler(
            self.trainer,
            event_name=Events.ITERATION_STARTED(every=1000),
            optimizer=optimizer,
        )

        # Attach logger to evaluator on test dataset
        for k in self.loader_keys:
            if k == "train":
                continue
            wandb_logger.attach_output_handler(
                self.evaluators[k],
                event_name=Events.EPOCH_COMPLETED,
                tag=k,
                metric_names=["rmse"],
                global_step_transform=lambda *_: self.trainer.state.iteration,
            )

        return wandb_logger
