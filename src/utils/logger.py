# src/utils/logger.py
import logging
import os


class Logger:
    def __init__(self, script_name: str):
        self.log_file = f"logs/{script_name}.log"
        self._initialize_logger()

    def _initialize_logger(self):
        """
        Initialize the logger

        :param :
        :return:
        """
        logging.basicConfig(
            filename=self.log_file,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(script_name)

    def log(self, message: str):
        """
        Write a log in the logger file

        :param message: Message to log
        :return:
        """
        self.logger.info(message)
