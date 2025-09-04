# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import Callable
import abc


class BaseJobExecutor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def submit_job(
        self,
        fn: Callable,
        **kwargs,
    ) -> Future:
        """
        Submit a job to the executor.

        :param fn: The function to execute.
        :param kwargs: The keyword arguments to pass to the function (e.g., priority).

        :return: A Future representing the execution of the function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """
        Clean up the executor, optionally waiting for currently running jobs to finish.

        :param wait: If True, wait for currently running jobs to finish before
        returning.
        """
        raise NotImplementedError
