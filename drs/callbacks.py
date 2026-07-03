from typing import TYPE_CHECKING, Optional
import time

if TYPE_CHECKING:
    from .engine import DRSEngine, SimulationResult
    from .variables import Variable

# TODO: Should this file be a part of the public api?
# TODO: Should we replace the "forward" system with a callback system on the levels?


class Callback:
    """Base class for DRS Engine callbacks.

    Subclass this to hook into the simulation lifecycle.
    """

    def on_simulation_start(self, engine: "DRSEngine") -> None:
        """Called before the simulation loop begins."""
        pass

    def on_step_start(self, engine: "DRSEngine") -> None:
        """Called at the beginning of each simulation step, after rates are zeroed and models evaluated."""
        pass

    def on_threshold(
        self, engine: "DRSEngine", trigger_var: "Variable", is_upper: bool
    ) -> None:
        """Called when a variable's threshold is the trigger for the next time step."""
        pass

    def on_deadlock(self, engine: "DRSEngine") -> None:
        """Called immediately before a DeadlockError is raised."""
        pass

    def on_complete(self, engine: "DRSEngine", result: "SimulationResult") -> None:
        """Called when the simulation loop has completely finished."""
        pass


class ProgressBarCallback(Callback):
    """A built-in callback that uses rich to display a progress bar for the simulation."""

    def __init__(self):
        try:
            from rich.progress import (
                Progress,
                TextColumn,
                BarColumn,
                TaskProgressColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
        except ImportError:
            raise ImportError(
                "The 'rich' package is required for the ProgressBarCallback. Install it with 'pip install rich'."
            )

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.task_id = None
        self.max_time = None

    def on_simulation_start(self, engine: "DRSEngine") -> None:
        self.progress.start()
        # the engine doesn't technically know its max_time until run() is called,
        # but max_time is passed to run(). We can extract it if we attach it to the engine temporarily,
        # or we just assume we'll get it from engine if we store it.
        # Wait, DRSEngine.run(max_time) takes max_time. So it's not known here unless we store it on the engine.
        # Let's assume engine._current_max_time is set before calling this.
        self.max_time = getattr(engine, "_current_max_time", None)

        if self.max_time:
            self.task_id = self.progress.add_task(
                "[cyan]Simulating...", total=self.max_time
            )
        else:
            self.task_id = self.progress.add_task("[cyan]Simulating...", total=None)

    def on_step_start(self, engine: "DRSEngine") -> None:
        if self.task_id is not None and self.max_time:
            self.progress.update(self.task_id, completed=engine.current_time)

    def on_complete(self, engine: "DRSEngine", result: "SimulationResult") -> None:
        if self.task_id is not None and self.max_time:
            self.progress.update(self.task_id, completed=self.max_time)
        self.progress.stop()

    def on_deadlock(self, engine: "DRSEngine") -> None:
        self.progress.stop()
