import json
import logging
import os
from collections import defaultdict
from queue import Queue
from threading import Thread
from typing import Any

import cloudpickle
import numpy as np
from numpy import typing as npt
from omegaconf import DictConfig
import omegaconf
from tabulate import tabulate
from safe_opax.rl import metrics as m

log = logging.getLogger("logger")

_SUMMARY_DEFAULT = "summary"


class TrainingLogger:
    def __init__(self, config: DictConfig):
        self._writters = []
        for writter in config.writters:
            if writter == "wandb":
                self._writters.append(WeightAndBiasesWritter(config))
            elif writter == "jsonl":
                self._writters.append(JsonlWritter(config.log_dir))
            elif writter == "tensorboard":
                self._writters.append(TensorboardXWritter(config.log_dir))
            elif writter == "stderr":
                self._writters.append(StdErrWritter())
            else:
                raise ValueError(f"Unknown writter: {writter}")

    def log(self, summary: dict[str, float], step: int):
        self._writter.log(summary, step)

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
    ):
        self._writter.log_video(images, step, name, fps)


class StdErrWritter:
    def __init__(self, logger_name: str = _SUMMARY_DEFAULT) -> None:
        self._logger = logging.getLogger(logger_name)

    def log(self, summary: dict[str, float], step: int):
        to_log = [[k, v] for k, v in summary.items()]
        self._logger.info(
            tabulate(to_log, headers=["Metric", "Value"], tablefmt="orgtbl")
        )

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
    ):
        pass


class JsonlWritter:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir

    def log(self, summary: dict[str, float], step: int):
        with open(os.path.join(self.log_dir, f"{_SUMMARY_DEFAULT}.jsonl"), "a") as file:
            file.write(json.dumps({"step": step, **summary}) + "\n")


class TensorboardXWritter:
    def __init__(self, log_dir) -> None:
        import tensorboardX

        self._writter = tensorboardX.FileWriter(log_dir)

    def log(self, summary: dict[str, float], step: int):
        for k, v in summary.items():
            self._writer.add_scalar(k, float(v), step)

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
        flush: bool = False,
    ):
        # (N, T, C, H, W)
        self._writer.add_video(
            name, np.array(images, copy=False).transpose([0, 1, 4, 2, 3]), step, fps=fps
        )
        if flush:
            self._writer.flush()


class WeightAndBiasesWritter:
    def __init__(self, config: DictConfig):
        import wandb

        wandb.init(
            project="safe-opax", resume=True, group=config.hydra.job.override_dirname
        )
        wandb.config = omegaconf.OmegaConf.to_container(config)
        self._handle = wandb

    def log(self, summary: dict[str, float], step: int):
        self._handle.log(summary, step=step)

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
    ):
        self._handle.log(
            {
                "video": self._handle.Video(
                    np.array(images, copy=False).transpose([0, 1, 4, 2, 3]),
                    fps=fps,
                    caption=name,
                )
            },
            step=step,
        )


class StateWriter:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.queue: Queue[bytes] = Queue(maxsize=5)
        self._thread = Thread(name="state_writer", target=self._worker)
        self._thread.start()

    def write(self, data: dict[str, Any]):
        state_bytes = cloudpickle.dumps(data)
        self.queue.put(state_bytes)
        # Lazily open up a thread and let it drain the work queue. Thread exits
        # when there's no more work to do.
        if not self._thread.is_alive():
            self._thread = Thread(name="state_writer", target=self._worker)
            self._thread.start()

    def _worker(self):
        while not self.queue.empty():
            state_bytes = self.queue.get(timeout=1)
            with open(os.path.join(self.log_dir, "state.pkl"), "wb") as f:
                f.write(state_bytes)
                self.queue.task_done()

    def close(self):
        self.queue.join()
        if self._thread.is_alive():
            self._thread.join()


class MetricsMonitor:
    def __init__(self):
        self._metrics = defaultdict(m.MetricsAccumulator)

    def __getitem__(self, item: str):
        return self._metrics[item]

    def __setitem__(self, key: str, value: float):
        self._metrics[key].update_state(value)

    def __str__(self) -> str:
        table = []
        for k, v in self._metrics.items():
            metrics = v.result
            table.append([k, metrics.mean, metrics.std, metrics.min, metrics.max])
        return tabulate(
            table,
            headers=["Metric", "Mean", "Std", "Min", "Max"],
            tablefmt="orgtbl",
        )
