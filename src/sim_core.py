# CORE LOGIC OF THE SIMULATION

import csv
import heapq
from pathlib import Path


def load_call_data_csv(filename): # function for loading call data from a CSV file
    call_data = []
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            call_data.append({
                "call_id": int(row["call_id"]),
                "arrival_time_min": float(row["arrival_time_min"]),
                "service_time_sec": float(row["service_time_sec"]),
            })
    return call_data

def simulate_call_center(call_data, n_agents, target_wait_sec=30):
                                    # how many agents can handle calls simultaneously
    if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents <= 0:
        raise ValueError("n_agents must be a positive integer")

    calls = sorted(call_data, key=lambda x: x["arrival_time_min"])
                   # sorts calls
                   # by arrival time in minutes to make sure they are processed in the correct order

    agent_free_times = [0.0] * n_agents
    # creates a list of the time all agents become free, in seconds

    heapq.heapify(agent_free_times)
    # creates a min-heap from the list of agent free times, so that we can efficiently find the earliest free agent

    records = []               # initializes a list to store the records of each call's processing details
    answered_within_target = 0 # initializes a counter for the number of calls answered within the target wait time
    total_wait = 0.0           # initializes a variable to accumulate the total wait time for all calls

    for call in calls:
        arrival_sec = call["arrival_time_min"] * 60.0
        service_sec = call["service_time_sec"]

        earliest_free = heapq.heappop(agent_free_times)
        # gets the agent that will be free the earliest (the root of the min-heap)

        start_sec = max(arrival_sec, earliest_free)
        wait_sec = start_sec - arrival_sec
        end_sec = start_sec + service_sec
        # call starts when both the call has arrived or when the earliest free agent is available (whichever is later)

        if wait_sec <= target_wait_sec:
            answered_within_target += 1
        # increments the counter if the call was answered within the target wait time

        total_wait += wait_sec # accumulates the total wait time for all calls
        heapq.heappush(agent_free_times, end_sec)
        # updates the agent's free time to when they will be done with this call and pushes it back into the min-heap

        records.append({
            "call_id": call["call_id"],
            "arrival_time_sec": arrival_sec,
            "service_time_sec": service_sec,
            "wait_time_sec": wait_sec,
            "start_time_sec": start_sec,
            "end_time_sec": end_sec,
            "answered_within_target": wait_sec <= target_wait_sec,
            "answered_within_30_sec": wait_sec <= 30,
        })

    n_calls = len(calls)
    summary = {
        "n_calls": n_calls,
        "n_agents": n_agents,
        "target_wait_sec": target_wait_sec,
        "answered_within_target": answered_within_target,
        "service_level": answered_within_target / n_calls if n_calls else 0.0,
        "average_wait_sec": total_wait / n_calls if n_calls else 0.0,
    }

    return summary, records

def find_min_agents(call_data, target_service_level=0.80, max_agents=50):
                               # 80% of calls answered within 30 seconds
                               # maximum number of agents to consider in search for the minimum required to meet the target service level
    for n_agents in range(1, max_agents + 1):
        summary, _ = simulate_call_center(call_data, n_agents)
        # returns only the summary of the simulation
        if summary["service_level"] >= target_service_level:
            return n_agents, summary
            # returns the minimum number of agents that meets the target service level and the corresponding summary of the simulation
    return None, None
    # returns None if no number of agents up to max_agents meets the target service level

if __name__ == "__main__":
    path = Path("results/csv/calls.csv")
    calls = load_call_data_csv(path)

    summary, _ = simulate_call_center(calls, n_agents=5)
    # runs the simulation with 5 agents and returns the summary of the simulation
    print(summary)

    best_agents, best_summary = find_min_agents(calls, target_service_level=0.80)
    # finds the minimum number of agents that meets the target service level of 80% and returns the corresponding summary of the simulation
    print(best_agents, best_summary)