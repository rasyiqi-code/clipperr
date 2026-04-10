import logging
import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_initialized = False


def setup_logging(level: int = logging.DEBUG) -> None:
    """Initialize the root logger once. Safe to call multiple times."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(level)

    # Silence verbose third-party libraries
    logging.getLogger("faster_whisper").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("numexpr").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Add File Handler for debugging on Windows
    try:
        import os
        log_file = os.path.join(os.path.expanduser("~"), "clipperr_log.txt")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)
        logging.info(f"Logging initialized. Log file at: {log_file}")
    except Exception as e:
        print(f"Failed to initialize file logger: {e}")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Initializes logging on first call."""
    setup_logging()
    return logging.getLogger(name)
