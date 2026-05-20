"""timing und token-messung fuer evaluations-laeufe

@measure(step_name) misst latenz (ms) und zaehlt input/output-tokens via tiktoken
ergebnisse landen in einer thread-sicheren liste
abrufbar via drain_measurements()

token-zaehlung ist naeherung
bei llm-calls mit response.usage werden die echten zahlen genommen (record_llm_usage)
sonst tiktoken-approximation
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


@dataclass
class Measurement:
    step_name: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


_measurements: list[Measurement] = []
_lock = Lock()


def _count_tokens(text: str | None) -> int:
    if not text:
        return 0
    if _ENCODING is None:
        return max(1, len(text) // 4)
    try:
        return len(_ENCODING.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def measure(step_name: str) -> Callable:
    """decorator misst laufzeit und (approximierte) tokens einer funktion
    token-zaehlung auf dem ersten string-argument (input)
    und dem rueckgabewert (output falls str)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            input_text = next((a for a in args if isinstance(a, str)), None)
            output_text = result if isinstance(result, str) else None

            m = Measurement(
                step_name=step_name,
                latency_ms=latency_ms,
                input_tokens=_count_tokens(input_text),
                output_tokens=_count_tokens(output_text),
            )
            with _lock:
                _measurements.append(m)
            return result

        return wrapper

    return decorator


def record_llm_usage(
    step_name: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    **extra: Any,
) -> None:
    """messung mit echten api-tokens einspeisen
    umgeht die tiktoken-approximation
    """
    m = Measurement(
        step_name=step_name,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        extra=dict(extra),
    )
    with _lock:
        _measurements.append(m)


def drain_measurements() -> list[Measurement]:
    """liefert die gesammelten messungen und leert den puffer"""
    with _lock:
        out = list(_measurements)
        _measurements.clear()
    return out


def reset() -> None:
    with _lock:
        _measurements.clear()
