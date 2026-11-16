import pytest

import math
from numpy.random import SeedSequence

from src.parallel_pool import run_parallel, _simulate_one
from src.parallel_numpy import run_numpy

from src.baseline import run_baseline
from src.data_gen import generate_call_data, save_call_data_csv
from src.sim_core import load_call_data_csv, simulate_call_center


def test_generated_call_data_round_trips_and_simulates(tmp_path, monkeypatch):
    call_data = generate_call_data(10, arrival_rate_per_minute=2, seed=7)
    monkeypatch.chdir(tmp_path)
    csv_path = "calls.csv"

    save_call_data_csv(call_data, csv_path)
    loaded_data = load_call_data_csv(csv_path)

    assert loaded_data == call_data
    assert loaded_data == generate_call_data(10, arrival_rate_per_minute=2, seed=7)

    summary, records = simulate_call_center(loaded_data, n_agents=3)
    assert summary["n_calls"] == len(records) == 10


def test_baseline_runs_when_imported_as_a_package(tmp_path):
    call_data = generate_call_data(10, arrival_rate_per_minute=2, seed=7)
    csv_path = tmp_path / "calls.csv"
    save_call_data_csv(call_data, csv_path)

    summary, records, min_agents, min_summary = run_baseline(csv_path, n_agents=3)

    assert summary["n_calls"] == len(records) == 10
    assert min_agents is not None
    assert min_summary["service_level"] >= 0.80


def test_call_records_report_both_configured_and_30_second_thresholds():
    call_data = [
        {"call_id": 1, "arrival_time_min": 0.0, "service_time_sec": 20.0},
        {"call_id": 2, "arrival_time_min": 0.0, "service_time_sec": 1.0},
    ]

    summary, records = simulate_call_center(call_data, n_agents=1, target_wait_sec=10)

    assert records[1]["wait_time_sec"] == 20.0
    assert records[1]["answered_within_target"] is False
    assert records[1]["answered_within_30_sec"] is True
    assert summary["answered_within_target"] == 1


@pytest.mark.parametrize("n_agents", [0, -1, 1.5, True])
def test_simulation_rejects_invalid_agent_counts(n_agents):
    call_data = [
        {"call_id": 1, "arrival_time_min": 0.0, "service_time_sec": 1.0},
    ]
    with pytest.raises(ValueError, match="n_agents must be a positive integer"):
        simulate_call_center(call_data, n_agents=n_agents)

def _run_sequential_reference(
    n_replications,
    n_calls,
    n_agents,
    arrival_rate,
    service_time,
    seed,
    target_wait,
):
    """
    Sequential "truth" for the replicated workload used by run_parallel/run_numpy.

    We must use the same SeedSequence.spawn(...) logic so results are comparable
    regardless of worker count.
    """
    child_seeds = SeedSequence(seed).spawn(n_replications)

    total_calls = 0
    total_answered = 0
    total_wait = 0.0

    for child in child_seeds:
        task = (
            int(child.generate_state(1)[0]),
            n_calls,
            arrival_rate,
            service_time,
            n_agents,
            target_wait,
        )
        calls, answered, wait = _simulate_one(task)
        total_calls += calls
        total_answered += answered
        total_wait += wait

    return {
        "n_calls": total_calls,
        "answered_within_target": total_answered,
        "service_level": total_answered / total_calls,
        "average_wait_sec": total_wait / total_calls,
    }


def _assert_matches_reference(ref, out):
    # Integers/counts must match exactly
    assert out["n_calls"] == ref["n_calls"]
    assert out["answered_within_target"] == ref["answered_within_target"]

    # Floats: allow tiny differences (order of operations / NumPy reductions)
    assert math.isclose(ref["service_level"], out["service_level"], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ref["average_wait_sec"], out["average_wait_sec"], rel_tol=1e-9, abs_tol=0.0)


def test_parallel_pool_matches_sequential_reference_and_is_worker_invariant():
    params = dict(
        n_replications=12,
        n_calls=200,
        n_agents=5,
        arrival_rate=2.0,
        service_time=180,
        seed=123,
        target_wait=30,
    )

    ref = _run_sequential_reference(**params)

    out_w1 = run_parallel(workers=1, **params)
    out_w2 = run_parallel(workers=2, **params)

    _assert_matches_reference(ref, out_w1)
    _assert_matches_reference(ref, out_w2)

    # Worker invariance: different worker counts must give the same result
    _assert_matches_reference(
        {
            "n_calls": out_w1["n_calls"],
            "answered_within_target": out_w1["answered_within_target"],
            "service_level": out_w1["service_level"],
            "average_wait_sec": out_w1["average_wait_sec"],
        },
        out_w2,
    )


def test_parallel_numpy_matches_sequential_reference_and_is_worker_invariant():
    params = dict(
        n_replications=12,
        n_calls=200,
        n_agents=5,
        arrival_rate=2.0,
        service_time=180,
        seed=123,
        target_wait=30,
    )

    ref = _run_sequential_reference(**params)

    out_w1 = run_numpy(workers=1, batch_size=4, **params)
    out_w2 = run_numpy(workers=2, batch_size=4, **params)

    _assert_matches_reference(ref, out_w1)
    _assert_matches_reference(ref, out_w2)

    # Worker invariance for the NumPy approach
    _assert_matches_reference(
        {
            "n_calls": out_w1["n_calls"],
            "answered_within_target": out_w1["answered_within_target"],
            "service_level": out_w1["service_level"],
            "average_wait_sec": out_w1["average_wait_sec"],
        },
        out_w2,
    )


@pytest.mark.parametrize("bad", [0, -1])
def test_parallel_versions_reject_non_positive_sizes(bad):
    with pytest.raises(ValueError):
        run_parallel(n_replications=bad)

    with pytest.raises(ValueError):
        run_numpy(n_replications=bad)
