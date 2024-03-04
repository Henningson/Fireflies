import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple


def extract_normalized_error(file_path: str) -> float:
    with open(file_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)

        # Get the first row
        first_row = next(csv_reader)

        # Access the third element
        normalized_error = float(first_row[2])
        return normalized_error


def extract_number_of_laser_rays(file_path: str) -> int:
    checkpoint_file = [file for file in os.listdir(file_path) if file.endswith(".tar")][
        0
    ]
    checkpoint = torch.load(os.path.join(file_path, checkpoint_file))
    return checkpoint["laser_rays"].shape[0]


def find_and_load_errors(base_path: str) -> Tuple[List[float], List[float]]:
    # List all folders in the given path
    folders = [
        folder
        for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder))
    ]
    folders.sort()

    normalized_mae_errors = []
    normalized_rsme_errors = []
    num_rays = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)

        mae_file_path = os.path.join(folder_path, "MAE.csv")
        rsme_file_path = os.path.join(folder_path, "RSME.csv")

        normalized_mae_errors.append(extract_normalized_error(mae_file_path))
        normalized_rsme_errors.append(extract_normalized_error(rsme_file_path))
        num_rays.append(extract_number_of_laser_rays(folder_path))

    return folders, normalized_mae_errors, normalized_rsme_errors, num_rays


def get_min_and_index(value_list: List[float]) -> Tuple[int, float]:
    minimum = min(value_list)
    return value_list.index(minimum), minimum


def find_best_performing_networks(scene_path, folders) -> None:
    for folder in folders:
        folder_base_path = os.path.join(scene_path, "optim", folder)
        checkpoint_folder, maes, rmses, rays = find_and_load_errors(folder_base_path)

        mae_min_index, mae_min = get_min_and_index(maes)
        rmse_min_index, rmse_min = get_min_and_index(rmses)

        print(f"Best performing networks for {folder}")
        print(
            f"{checkpoint_folder[mae_min_index]} - Rays; {rays[mae_min_index]} - MAE: {mae_min:.5f}"
        )
        print(
            f"{checkpoint_folder[rmse_min_index]} - Rays; {rays[rmse_min_index]} - RMSE: {rmse_min:.5f}"
        )


def generate_performance_plot(scene_path, folders):

    fig, axs = plt.subplots(2)
    fig.suptitle("MAE and RSME curves")

    for folder in folders:
        folder_base_path = os.path.join(scene_path, "optim", folder)
        _, maes, rmses, num_rays = find_and_load_errors(folder_base_path)

        axs[0].plot(num_rays, maes, label=folder, lw=10, solid_capstyle="round")
        axs[1].plot(num_rays, rmses, label=folder, lw=10, solid_capstyle="round")

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    vocalfold_base_path = "scenes/Vocalfold"
    realcolon_base_path = "scenes/RealColon"

    folders = [
        "POISSON_LR",
        "RANDOM_LR",
        "GRID_LR",
        "SMARTY_LR",
    ]
    find_best_performing_networks(vocalfold_base_path, folders)
    generate_performance_plot(vocalfold_base_path, folders)
