# FOR CREATING SYNTHETIC CALL DATA FOR SIMULATION

import csv
from pathlib import Path

import numpy as np

def generate_arrival_times(n_calls, arrival_rate_per_minute, seed=42):
    rng = np.random.default_rng(seed) # with set seed so that results are reproducible
    interarrival_times = rng.exponential(scale=1 / arrival_rate_per_minute, size=n_calls)
                                               # average gap per minute
                                               # gaps vary randomly (gaps are exponentially distributed)
    return np.cumsum(interarrival_times)       # makes a list of the arrival times by adding the gaps

def generate_service_times(n_calls, mean_service_time_seconds=180, seed=42):
    rng = np.random.default_rng(seed) # with set seed so that results are reproducible
    return rng.exponential(scale=mean_service_time_seconds, size=n_calls)
                           # sets the average service time in seconds (3 minutes)
                           # service times vary randomly (service times are exponentially distributed)

def generate_call_data(n_calls, arrival_rate_per_minute, mean_service_time_seconds=180, seed=42):
    arrival_times = generate_arrival_times(n_calls, arrival_rate_per_minute, seed=seed)
    service_times = generate_service_times(n_calls, mean_service_time_seconds, seed=seed + 1)

    return [
        {
            "call_id": i,
            "arrival_time_min": float(arrival_times[i]),
            "service_time_sec": float(service_times[i]),
        }
        for i in range(n_calls) # returns a list of dictionaries, each representing a call with its ID, arrival time in minutes, and service time in seconds
    ]

def save_call_data_csv(call_data, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["call_id", "arrival_time_min", "service_time_sec"])
        writer.writeheader()
        writer.writerows(call_data) # writes the data to the file in CSV format with headers for call ID, arrival time in minutes, and service time in seconds

if __name__ == "__main__": # if data_gen.py is run, it will generate synthetic call data and save it to a CSV file
    data = generate_call_data(
        n_calls=100,
        arrival_rate_per_minute=2,
        mean_service_time_seconds=180,
        seed=42
    )
    save_call_data_csv(data, "results/csv/calls.csv") # saves the generated call data to a CSV file in the specified directory
    print(data[:5])