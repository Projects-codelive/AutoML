import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# ── Root logs folder (AutoML/logs/) ───────────────────────────────────────────
LOGS_ROOT = Path(__file__).resolve().parent.parent / "logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

# ── Generate ONE timestamp for the entire session ──────────────────────────────
# This is created ONCE when logger.py is first imported.
# Every module that calls get_logger() in the same run shares this filename.
# Result: all logs from one pipeline run go into one file e.g. 2026-06-18_16-49-14.log
_SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_SESSION_LOG_FILE = LOGS_ROOT / f"{_SESSION_TIMESTAMP}.log"


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module.

    Log output goes to:
      - Console (INFO and above)
      - logs/YYYY-MM-DD_HH-MM-SS.log (DEBUG and above, shared across all modules)

    No subfolders. No separate master file.
    All modules in one pipeline run share the same timestamped log file.

    Args:
        module_name: Name of the module e.g. "task_detection", "eda"

    Returns:
        A configured logging.Logger instance
    """

    # ── 1. Get or create logger by module name ─────────────────────────────────
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # ── 2. Guard: don't add handlers again if already configured ──────────────
    # This matters when get_logger() is called multiple times in the same run
    if logger.handlers:
        return logger

    # ── 3. Log format ──────────────────────────────────────────────────────────
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # ── 4. Handler A: Console — INFO and above ─────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── 5. Handler B: Timestamped log file — DEBUG and above ───────────────────
    # Uses the session timestamp so every run creates a NEW file.
    # RotatingFileHandler still protects against a single file growing too large.
    file_handler = RotatingFileHandler(
        filename=_SESSION_LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB cap per file
        backupCount=3,                # if it somehow hits 10MB: .log → .log.1 etc.
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ── 6. Attach handlers ─────────────────────────────────────────────────────
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger