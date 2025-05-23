# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from logging import Logger
import logging
import os


def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return (
        f"{color}[%(asctime)s] LMCache %(levelname)s:{reset} %(message)s "
        f"{underline}(%(filename)s:%(lineno)d:%(name)s){reset}"
    )


class CustomFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: build_format(grey),
        logging.INFO: build_format(green),
        logging.WARNING: build_format(yellow),
        logging.ERROR: build_format(red),
        logging.CRITICAL: build_format(bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_log_level() -> int:
    """
    Try to read LMCACHE_LOG_LEVEL from environment variables.
    Could be:
    - DEBUG
    - INFO
    - WARNING
    - ERROR
    - CRITICAL

    If not found, defaults to INFO.
    """
    log_level = os.getenv("LMCACHE_LOG_LEVEL", "INFO").upper()
    return getattr(logging, log_level, logging.INFO)


def init_logger(name: str) -> Logger:
    # Get the logger
    logger = logging.getLogger(name)

    # Clear any existing handlers
    logger.handlers.clear()

    # Prevent propagation to parent loggers
    logger.propagate = False

    # Add our custom handler
    ch = logging.StreamHandler()
    ch.setLevel(get_log_level())
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    logger.setLevel(get_log_level())
    return logger


if __name__ == "__main__":
    logger = init_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

# import logging
# from logging import Logger
#
# logging.basicConfig(
#    format="\033[33m%(levelname)s LMCache: \033[0m%(message)s "
#    "[%(asctime)s] -- %(pathname)s:%(lineno)d",
#    level=logging.INFO,
# )
#
#
# def init_logger(name: str) -> Logger:
#    logger = logging.getLogger(name)
#    logger.setLevel(logging.DEBUG)
#    return logger
