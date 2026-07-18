import logging
from datetime import datetime
from typing import Any, Dict

try:
    from ..config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, LOG_DIR

except ImportError:
    from app.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, LOG_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # skip if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # console handler with simple formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # structured logging handling
    try:
        from pythonjsonlogger import jsonlogger
        
        class JSONFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
                super().add_fields(log_record, record, message_dict)
                log_record['timestamp'] = datetime.utcnow().isoformat()
                log_record['level'] = record.levelname
                log_record['logger'] = record.name
                log_record['module'] = record.module
                log_record['function'] = record.funcName
                log_record['line'] = record.lineno
        
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    except ImportError:
        # Fallback to simple file logging if pythonjsonlogger not available
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
        file_formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# create root logger
logger = get_logger("gold_price_forecast")