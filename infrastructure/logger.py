"""
Advanced Logging Infrastructure for Quant Pipeline
Structured logging with multiple handlers, correlation IDs, and audit trails
"""

import json
import logging
import sys
import traceback
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add strategy context if present
        if hasattr(record, "strategy"):
            log_data["strategy"] = record.strategy

        # Add custom fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data)


class AuditLogger:
    """Dedicated audit logger for regulatory compliance"""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)

        # Audit logs are append-only and never rotated (for compliance)
        audit_file = self.log_dir / f'audit_{datetime.now().strftime("%Y%m%d")}.jsonl'
        handler = logging.FileHandler(audit_file)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

    def log_trade(self, order_id: str, action: str, details: Dict[str, Any]):
        """Log trade execution for audit trail"""
        self.logger.info(
            f"Trade {action}: {order_id}",
            extra={
                "extra_fields": {
                    "event_type": "trade",
                    "order_id": order_id,
                    "action": action,
                    "details": details,
                    "timestamp_utc": datetime.utcnow().isoformat(),
                }
            },
        )

    def log_risk_event(self, event_type: str, details: Dict[str, Any]):
        """Log risk management events"""
        self.logger.warning(
            f"Risk event: {event_type}",
            extra={
                "extra_fields": {
                    "event_type": "risk",
                    "risk_event": event_type,
                    "details": details,
                    "timestamp_utc": datetime.utcnow().isoformat(),
                }
            },
        )

    def log_model_change(self, model_id: str, version: str, details: Dict[str, Any]):
        """Log model deployment/changes"""
        self.logger.info(
            f"Model change: {model_id} v{version}",
            extra={
                "extra_fields": {
                    "event_type": "model_deployment",
                    "model_id": model_id,
                    "version": version,
                    "details": details,
                    "timestamp_utc": datetime.utcnow().isoformat(),
                }
            },
        )


class QuantLogger:
    """Main logging infrastructure for quant pipeline"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.log_dir = Path(__file__).parent.parent / "monitoring" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Main application logger
        self.logger = logging.getLogger("quant")
        self.logger.setLevel(logging.DEBUG)

        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler with structured JSON logs
        file_handler = logging.FileHandler(
            self.log_dir / f'quant_{datetime.now().strftime("%Y%m%d")}.jsonl'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)

        # Error-only handler
        error_handler = logging.FileHandler(self.log_dir / "errors.jsonl")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)

        # Audit logger
        self.audit = AuditLogger(self.log_dir / "audit")

        self._initialized = True

    def get_logger(self, name: str) -> logging.Logger:
        """Get a child logger with the given name"""
        return self.logger.getChild(name)

    def with_correlation_id(
        self, correlation_id: Optional[str] = None
    ) -> "CorrelationLogger":
        """Create a logger with correlation ID for request tracing"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        return CorrelationLogger(self.logger, correlation_id)


class CorrelationLogger:
    """Logger wrapper that adds correlation ID to all logs"""

    def __init__(self, logger: logging.Logger, correlation_id: str):
        self.logger = logger
        self.correlation_id = correlation_id

    def _add_correlation(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["correlation_id"] = self.correlation_id
        return kwargs

    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, **self._add_correlation(kwargs))

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, **self._add_correlation(kwargs))

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, **self._add_correlation(kwargs))

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, **self._add_correlation(kwargs))

    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, **self._add_correlation(kwargs))


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            correlation_id = str(uuid.uuid4())

            logger.info(
                f"Starting {func.__name__}", extra={"correlation_id": correlation_id}
            )

            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start).total_seconds()

                logger.info(
                    f"Completed {func.__name__} in {duration:.2f}s",
                    extra={
                        "correlation_id": correlation_id,
                        "extra_fields": {"duration_seconds": duration},
                    },
                )
                return result

            except Exception as e:
                duration = (datetime.now() - start).total_seconds()
                logger.error(
                    f"Failed {func.__name__} after {duration:.2f}s: {str(e)}",
                    extra={"correlation_id": correlation_id},
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# Global logger instance
quant_logger = QuantLogger()
