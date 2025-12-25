import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


def get_logger(
    name: str = "ml-pipeline",
    level=logging.INFO,
    log_file: Path = LOG_FILE
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # duplicate

    if log_file.exists():
        log_file.unlink()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] "
        "[%(filename)s:%(lineno)d] - %(message)s"
    )

    # 🔹 Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 🔹 File handler (rotating)
    file_handler = RotatingFileHandler(
        log_file,
        mode="w",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=1
    )
    file_handler.setFormatter(formatter)

    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


LOG = get_logger()
