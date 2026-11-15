# RUNNING THE SIMULATION

from pathlib import Path

if __package__:
    from .sim_core import load_call_data_csv, simulate_call_center, find_min_agents
else:
    from sim_core import load_call_data_csv, simulate_call_center, find_min_agents

DEFAULT_DATA_PATH = Path("results/csv/calls.csv") # sets the default path to the CSV file containing the call data

def run_baseline(data_path=DEFAULT_DATA_PATH, n_agents=5, target_service_level=0.80):
    calls = load_call_data_csv(data_path)

    summary, records = simulate_call_center(calls, n_agents=n_agents)
    # run with one fixed staffing level

    min_agents, min_summary = find_min_agents(calls, target_service_level=target_service_level)
    # find the minimum number of agents that meets the target

    return summary, records, min_agents, min_summary

if __name__ == "__main__":
    summary, records, min_agents, min_summary = run_baseline()

    print("Baseline run:")
    print(summary)

    print("\nMinimum agents meeting target:")
    print(min_agents)
    print(min_summary)