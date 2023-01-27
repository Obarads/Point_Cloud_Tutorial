import logging
import yaml
import argparse

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class LoggingUtils:
    @staticmethod
    def create_logger(
        process_name: str, level=logging.DEBUG, formatter=FORMATTER
    ):
        logger = logging.getLogger(process_name)
        logger.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    @staticmethod
    def add_file_handler(
        log_file_path: str,
        logger: logging.Logger,
        level=logging.DEBUG,
        formatter=FORMATTER,
    ):
        fh = logging.FileHandler(log_file_path, mode="w")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def save_args(output_file_path: str, args: argparse.Namespace):
    args_dict = vars(args)
    with open(output_file_path, "w") as f:
        yaml.dump(args_dict, f, encoding="utf-8", allow_unicode=True)
