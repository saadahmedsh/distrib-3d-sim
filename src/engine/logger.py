import logging
import sys
from typing import Optional


class DistributedLogger:
    """
    A professional, structured logger designed to track tensor flows and distributed execution.

    This logger ensures that standard log formats are adhered to, avoiding emojis or unstructured
    filler. It supports adding a rank-specific prefix to differentiate messages originating from
    different virtual nodes in the simulation.
    """

    def __init__(self, rank: Optional[int] = None, name: str = "distrib-3d-sim") -> None:
        """
        Initializes the DistributedLogger.

        Args:
            rank: The rank of the current process, used as a prefix in log messages if provided.
            name: The name of the logger instance.
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.rank: Optional[int] = rank

        if not self.logger.handlers:
            handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter: logging.Formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _format_message(self, message: str) -> str:
        """
        Formats the message by prepending the rank if it is available.

        Args:
            message: The core log message.

        Returns:
            The formatted log message including the rank prefix.
        """
        if self.rank is not None:
            return f"[Rank {self.rank}] {message}"
        return message

    def info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message: The message to be logged.
        """
        formatted_message: str = self._format_message(message)
        self.logger.info(formatted_message)

    def debug(self, message: str) -> None:
        """
        Logs a debug message.

        Args:
            message: The message to be logged.
        """
        formatted_message: str = self._format_message(message)
        self.logger.debug(formatted_message)

    def error(self, message: str) -> None:
        """
        Logs an error message.

        Args:
            message: The message to be logged.
        """
        formatted_message: str = self._format_message(message)
        self.logger.error(formatted_message)
