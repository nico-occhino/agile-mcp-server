import time, functools
from workflow.logging import log

def instrumented(tool_name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            log.info("tool_call_start", tool=tool_name)
            try:
                result = fn(*args, **kwargs)
                duration = time.monotonic() - start
                log.info(
                    "tool_call_end",
                    tool=tool_name,
                    duration_s=round(duration, 3),
                    confidence=result.get("confidence") if isinstance(result, dict) else None,
                    confidence_level=result.get("confidence_level") if isinstance(result, dict) else None,
                    ok=True,
                )
                return result
            except Exception as exc:
                duration = time.monotonic() - start
                log.error(
                    "tool_call_error",
                    tool=tool_name,
                    duration_s=round(duration, 3),
                    error_type=type(exc).__name__,
                    ok=False,
                )
                raise
        return wrapper
    return decorator
