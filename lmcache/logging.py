import logging
from logging import Logger

logging.basicConfig(
        format='\033[33m%(levelname)s LMCache: \033[0m%(message)s [%(asctime)s.%(msecs)03d]', 
        level=logging.INFO,
    )

def init_logger(name: str) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
