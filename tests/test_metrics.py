"""Analytical queueing metrics for the call-centre simulation.

The functions in this module use the steady-state M/M/c model (Poisson
arrivals, exponential service times, and c identical agents). Arrival rates
are expressed in calls per minute; mean service times and waiting-time targets
are expressed in seconds.
"""

from __future__ import annotations

import math


def _validate_queue_inputs(
    arrival_rate_per_minute: float,
    mean_service_time_seconds: float,
    n_agents: int,
) -> tuple[float, float, int]:
    """Validate inputs and return (arrival rate, service rate, agent count).

    The service rate is converted to calls per minute per agent so that it
    uses the same time unit as the arrival rate.
    """
    if isinstance(arrival_rate_per_minute, bool) or not isinstance(
        arrival_rate_per_minute, (int, float)
    ):
        raise ValueError("arrival_rate_per_minute must be a finite non-negative number")
    if not math.isfinite(arrival_rate_per_minute) or arrival_rate_per_minute < 0:
        raise ValueError("arrival_rate_per_minute must be a finite non-negative number")

    if isinstance(mean_service_time_seconds, bool) or not isinstance(
        mean_service_time_seconds, (int, float)
    ):
        raise ValueError("mean_service_time_seconds must be a finite positive number")
    if not math.isfinite(mean_service_time_seconds) or mean_service_time_seconds <= 0:
        raise ValueError("mean_service_time_seconds must be a finite positive number")

    if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents <= 0:
        raise ValueError("n_agents must be a positive integer")

    service_rate_per_minute = 60.0 / mean_service_time_seconds
    return float(arrival_rate_per_minute), service_rate_per_minute, n_agents


def erlang_c_wait_probability(
    arrival_rate_per_minute: float,
    mean_service_time_seconds: float,
    n_agents: int,
) -> float:
    """Return the steady-state probability that an arriving call must wait.

    This is the Erlang C probability for an M/M/c queue. A stable queue is
    required, meaning the total service capacity must exceed the arrival rate:
    arrival_rate_per_minute < n_agents * service_rate_per_minute.

    The calculation uses the recursive Erlang B formula rather than
    factorials. This avoids unnecessarily large intermediate values.
    """
    arrival_rate, service_rate, n_agents = _validate_queue_inputs(
        arrival_rate_per_minute, mean_service_time_seconds, n_agents
    )

    if arrival_rate == 0:
        return 0.0

    offered_load = arrival_rate / service_rate
    utilisation = offered_load / n_agents
    if utilisation >= 1.0:
        raise ValueError(
            "The M/M/c queue is unstable: total agent service capacity must "
            "exceed the arrival rate"
        )

    # Erlang B recursion: B_k = (a * B_(k-1)) / (k + a * B_(k-1)).
    # It is numerically more stable than directly evaluating factorials.
    erlang_b = 1.0
    for servers in range(1, n_agents + 1):
        erlang_b = (offered_load * erlang_b) / (
            servers + offered_load * erlang_b
        )

    # Convert the Erlang B blocking probability into the Erlang C
    # probability that an arriving call has to wait for an agent.
    return erlang_b / (1.0 - utilisation + utilisation * erlang_b)


def erlang_c_service_level(
    arrival_rate_per_minute: float,
    mean_service_time_seconds: float,
    n_agents: int,
    target_wait_sec: float = 30.0,
) -> float:
    """Return the theoretical share of calls answered within a wait target.

    For an M/M/c queue, the probability that a call waits longer than t is
    ErlangC * exp(-(c * mu - lambda) * t). The target is converted from seconds
    to minutes before applying the formula.
    """
    arrival_rate, service_rate, n_agents = _validate_queue_inputs(
        arrival_rate_per_minute, mean_service_time_seconds, n_agents
    )

    if isinstance(target_wait_sec, bool) or not isinstance(target_wait_sec, (int, float)):
        raise ValueError("target_wait_sec must be a finite non-negative number")
    if not math.isfinite(target_wait_sec) or target_wait_sec < 0:
        raise ValueError("target_wait_sec must be a finite non-negative number")

    probability_wait = erlang_c_wait_probability(
        arrival_rate, mean_service_time_seconds, n_agents
    )
    spare_capacity_per_minute = n_agents * service_rate - arrival_rate
    target_wait_minutes = target_wait_sec / 60.0

    probability_exceeding_target = probability_wait * math.exp(
        -spare_capacity_per_minute * target_wait_minutes
    )
    return 1.0 - probability_exceeding_target


def find_minimum_agents_erlang_c(
    arrival_rate_per_minute: float,
    mean_service_time_seconds: float,
    target_service_level: float = 0.80,
    target_wait_sec: float = 30.0,
    max_agents: int = 50,
) -> int | None:
    """Find the fewest agents meeting a service-level target under M/M/c.

    Returns the minimum stable agent count that reaches the requested
    service level, or ``None`` when no count up to ``max_agents`` is enough.
    This is an analytical steady-state estimate, not a substitute for the
    finite-sample simulation or a model with time-varying arrivals.
    """
    # Validate the common queue parameters before starting the search.
    arrival_rate, service_rate, _ = _validate_queue_inputs(
        arrival_rate_per_minute, mean_service_time_seconds, 1
    )

    if isinstance(target_service_level, bool) or not isinstance(
        target_service_level, (int, float)
    ):
        raise ValueError("target_service_level must be between 0 and 1")
    if (
        not math.isfinite(target_service_level)
        or not 0.0 <= target_service_level <= 1.0
    ):
        raise ValueError("target_service_level must be between 0 and 1")

    if isinstance(max_agents, bool) or not isinstance(max_agents, int) or max_agents <= 0:
        raise ValueError("max_agents must be a positive integer")

    if isinstance(target_wait_sec, bool) or not isinstance(target_wait_sec, (int, float)):
        raise ValueError("target_wait_sec must be a finite non-negative number")
    if not math.isfinite(target_wait_sec) or target_wait_sec < 0:
        raise ValueError("target_wait_sec must be a finite non-negative number")

    offered_load = arrival_rate / service_rate

    # An M/M/c queue only has a steady state when c is strictly greater
    # than the offered load. Start at the first stable integer staffing level.
    first_stable_agent_count = max(1, math.floor(offered_load) + 1)
    for n_agents in range(first_stable_agent_count, max_agents + 1):
        service_level = erlang_c_service_level(
            arrival_rate,
            mean_service_time_seconds,
            n_agents,
            target_wait_sec,
        )
        if service_level >= target_service_level:
            return n_agents

    return None

