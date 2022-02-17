import os
from datetime import timedelta
from pathlib import Path
from typing import (Any, Generator, Iterable, List, Optional, Sequence,
                    TypeVar, Union)

from rich import print
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.text import Text
from torch.utils.data.dataloader import DataLoader

PathLike = Union[str, Path]


def repeat(loader: DataLoader) -> Generator[Any, None, None]:
    while True:
        for batch in loader:
            yield batch


def check_exist(paths: Sequence[Path]) -> List[Path]:
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"{path} does not exist!")
    return paths


T = TypeVar("T")


class TimeProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining

        elapsed_text = total_text = "-:--:--.---"

        if elapsed is not None:
            elapsed_text = str(timedelta(milliseconds=int(elapsed * 1000)))[:-3]

            if remaining is not None:
                total_text = str(
                    timedelta(milliseconds=int(elapsed * 1000) + int(remaining * 1000))
                )[:-3]

        return Text(f"[{elapsed_text} / {total_text}]", style="progress.remaining")


class StepProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:

        return Text(
            f"{int(task.completed)} / {int(task.total)}", style="progress.percentage"
        )


class SpeedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed

        if elapsed is None or elapsed == 0:
            return Text("", style="progress.elapsed")

        it_per_sec = task.completed / elapsed

        if it_per_sec > 1:
            return Text(f"{it_per_sec:3.3f}it/s", style="progress.elapsed")
        elif it_per_sec > 0:
            return Text(f"{1/it_per_sec:3.3f}s/it", style="progress.elapsed")
        else:
            return Text("", style="progress.elapsed")


def track(
    sequence: Union[Sequence[T], Iterable[T]],
    description: str = "Working...",
    total: Optional[float] = None,
    transient: bool = False,
) -> Iterable[T]:

    if isinstance(sequence, Sequence):
        total = total or len(sequence)

    columns: List["ProgressColumn"] = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(
            bar_width=None,
            style="bar.back",
            complete_style="bar.complete",
            finished_style="bar.finished",
            pulse_style="bar.pulse",
        ),
        StepProgressColumn(),
        TimeProgressColumn(),
        SpeedColumn(),
    ]
    progress = Progress(
        *columns,
        auto_refresh=True,
        console=None,
        transient=transient,
        get_time=None,
        refresh_per_second=10,
        disable=False,
    )

    with progress:
        yield from progress.track(
            sequence,
            total=total,
            description=description,
            update_period=0.1,
        )
