#!/usr/bin/env python
"""Circuit breaker pattern implementation for preventing cascading failures."""

import time
from enum import Enum
from typing import Callable, Any


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    After threshold failures, switches to OPEN state and fails fast for timeout period.
    After timeout, enters HALF_OPEN state to test if service recovered.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        name: str = "circuit_breaker",
    ):
        """
        Initialize circuit breaker.

        Parameters
        ----------
        failure_threshold : int
            Number of consecutive failures before opening circuit
        timeout : float
            Seconds to wait before attempting recovery (HALF_OPEN)
        name : str
            Circuit breaker identifier for logging
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.timeout

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Parameters
        ----------
        func : callable
            Function to execute
        *args, **kwargs
            Arguments to pass to function

        Returns
        -------
        Any
            Function return value

        Raises
        ------
        CircuitBreakerError
            If circuit is OPEN and timeout has not elapsed
        """
        # Check current state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                print(f"Circuit breaker '{self.name}': Attempting recovery (HALF_OPEN)")
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Failing fast. Retry after {self.timeout}s timeout."
                )

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Success: reset failure count and close circuit
            if self.state == CircuitState.HALF_OPEN:
                print(f"Circuit breaker '{self.name}': Service recovered (CLOSED)")

            self.failure_count = 0
            self.state = CircuitState.CLOSED
            self.last_failure_time = None

            return result

        except Exception as e:
            # Failure: increment count and potentially open circuit
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold or self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                print(
                    f"Circuit breaker '{self.name}': OPENED after {self.failure_count} failures. "
                    f"Timeout: {self.timeout}s"
                )

            raise e

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        print(f"Circuit breaker '{self.name}': Manually reset to CLOSED")

    def get_stats(self) -> dict[str, Any]:
        """
        Get circuit breaker statistics.

        Returns
        -------
        dict
            Statistics including state, failure_count, last_failure_time
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "timeout": self.timeout,
        }
