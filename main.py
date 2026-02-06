import os
import json
from config import *
from data.dataset import get_data_loaders
from experiments.run_experiment import run_experiment_for_eta
from analysis.visualize import plot_results

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_loader, val_loader, val_labels = get_data_loaders(DATASET_PATH, DEVICE)
    results = {}
    for eta in ETA_VALUES:
        metrics = run_experiment_for_eta(eta, train_loader, val_loader, val_labels, RESULTS_DIR)
        results[str(eta)] = metrics
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_results(os.path.join(RESULTS_DIR, "results.json"), os.path.join(RESULTS_DIR, "plot.png"))

if __name__ == "__main__":
    main()