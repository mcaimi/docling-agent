#!/usr/bin/env python
"""Unit tests for circuit breaker pattern."""

import time
import pytest
from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""

    def test_initial_state(self):
        """Circuit breaker should start in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_successful_call(self):
        """Successful calls should keep circuit CLOSED."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)

        def success_func():
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_failure_increments_count(self):
        """Failed calls should increment failure count."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)

        def failing_func():
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

        # Second failure
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.failure_count == 2
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        """Circuit should OPEN after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)

        def failing_func():
            raise ValueError("test error")

        # Reach threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_fails_fast_when_open(self):
        """Circuit should fail fast when OPEN."""
        cb = CircuitBreaker(failure_threshold=2, timeout=1.0)

        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Should fail fast without calling function
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

    def test_half_open_after_timeout(self):
        """Circuit should enter HALF_OPEN state after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.5)

        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.6)

        # Should attempt recovery (will fail and reopen)
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Should have transitioned to HALF_OPEN then back to OPEN
        assert cb.state == CircuitState.OPEN

    def test_recovery_success(self):
        """Circuit should close after successful call in HALF_OPEN state."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.5)

        def failing_func():
            raise ValueError("test error")

        def success_func():
            return "recovered"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.6)

        # Successful call should close circuit
        result = cb.call(success_func)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_manual_reset(self):
        """Manual reset should close circuit and clear failures."""
        cb = CircuitBreaker(failure_threshold=2, timeout=1.0)

        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 2

        # Manual reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_get_stats(self):
        """Stats should reflect current circuit state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=10.0, name="test_cb")

        stats = cb.get_stats()
        assert stats["name"] == "test_cb"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["failure_threshold"] == 3
        assert stats["timeout"] == 10.0

    def test_call_with_args_kwargs(self):
        """Circuit breaker should pass arguments correctly."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)

        def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = cb.call(func_with_args, "x", "y", c="z")
        assert result == "x-y-z"
        assert cb.state == CircuitState.CLOSED
