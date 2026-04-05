# File: app/runtime/background.py
# Purpose: Execute non-critical study tasks in the background.
# License: GPL-3.0-or-later
"""Background task execution for non-blocking side work."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BackgroundTaskManager:
    """Small wrapper around a thread pool for sidecar work."""

    max_workers: int = 4
    _executor: ThreadPoolExecutor = field(init=False, repr=False)
    _futures: list[Future[Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="clae-bg")

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        """Schedule a task and remember it for later draining."""

        future = self._executor.submit(fn, *args, **kwargs)
        self._futures.append(future)
        return future

    def drain(self) -> None:
        """Wait for all scheduled tasks to complete and surface the first error."""

        if not self._futures:
            return
        done, _ = wait(self._futures)
        self._futures.clear()
        for future in done:
            future.result()

    def close(self) -> None:
        """Shut down the worker pool."""

        self._executor.shutdown(wait=True, cancel_futures=False)
