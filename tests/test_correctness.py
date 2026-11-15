import pytest

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
