from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import sys
import time
from config import *
from ucimlrepo import fetch_ucirepo
import random
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from process_dataset import *
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class DifferentiableDecisionNode(nn.Module):
    def __init__(self):
        super(DifferentiableDecisionNode, self).__init__()
        # Parameter for the decision threshold
        self.decision = nn.Parameter(torch.randn(1))
        # self.weight=nn.Parameter(torch,randn(1))

    def forward(self, x):
        return torch.sigmoid(self.decision - x)


class DifferentiableDecisionTree(nn.Module):

    def __init__(self, depth, num_classes, ranked_features_indice):
        super(DifferentiableDecisionTree, self).__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.ranked_features_indice = ranked_features_indice
        self.nodes = nn.ModuleList(
            [DifferentiableDecisionNode() for _ in range(2**depth - 1)]
        )
        # Adjusting leaf values to accommodate class scores
        self.leaf_values = nn.Parameter(torch.randn(2**depth, num_classes))

    def forward(self, x):
        batch_size, num_features = x.shape
        path_probabilities = torch.ones(batch_size, 2**self.depth, device=x.device)
        node_index = 0
        x = x[:, self.ranked_features_indice]

        for level in range(self.depth):
            level_start = 2**level - 1
            parent_probabilities = path_probabilities.clone()

            indices = torch.arange(2**level, device=x.device)
            node_indices = level_start + indices

            decisions = torch.stack(
                [
                    self.nodes[idx](x[:, idx % num_features]).squeeze()
                    for idx in node_indices
                ],
                dim=1,
            )

            left_children = indices * 2
            right_children = left_children + 1

            path_probabilities[:, left_children] = (
                parent_probabilities[:, indices] * decisions
            )
            path_probabilities[:, right_children] = parent_probabilities[:, indices] * (
                1 - decisions
            )

        output = torch.matmul(path_probabilities, self.leaf_values)
        return output


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate log softmax and get probabilities for each class
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        # Get the log probability of the correct class (p_t) for each sample
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        probs = target_log_probs.exp()
        # Focal Loss computation: FL = -α * (1 - p_t)^γ * log(p_t)
        focal_loss = -self.alpha * (1 - probs) ** self.gamma * target_log_probs

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def load_dataset(name):
    X = pd.read_csv(f"datasets/{name}/X_{name}.csv")
    y = pd.read_csv(f"datasets/{name}/y_{name}.csv").squeeze()
    return X, y


def evaluate_model(model, X, y, num_classes):
    with torch.no_grad():
        start_time = time.time()
        predictions = model(X)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_classes = probabilities.argmax(dim=1)
        y_true = y.cpu().numpy()
        y_pred = predicted_classes.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_time = time.time() - start_time
        f1_macro = f1_score(y_true, y_pred, average="macro")

        return accuracy, f1_macro, accuracy_time


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment_trials(name):
    all_trials_results = []

    # Loop over each seed for multiple trials
    for trial, seed in enumerate(seed_list):
        dataset_dict = get_preprocessed_dataset(
            name,
            random_seed=seed,
            config=config_training,
            verbosity=0,
        )
        X_train_scaled = dataset_dict["X_train"].values
        X_val_scaled = dataset_dict["X_valid"].values
        X_test_scaled = dataset_dict["X_test"].values
        y_train = dataset_dict["y_train"].values
        y_val = dataset_dict["y_valid"].values
        y_test = dataset_dict["y_test"].values
        y_train_full = np.concatenate((y_train, y_val))
        y = np.concatenate((y_train_full, y_test))

        print(f"dataset name is: {name} ")
        print(f"Running trial {trial + 1} with seed {seed}", flush=True)
        # # Split the data into training and testing sets
        # print("Unique classes in y_train:", np.unique(y_train))
        # print("Unique classes in y_test:", np.unique(y_test))

        y_train_full = torch.tensor(y_train_full, dtype=torch.long)
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
        X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        mi = mutual_info_classif(X_train, y_train, random_state=seed)
        ranked_indices = np.argsort(mi)[::-1].copy()

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device, flush=True)
        # Move data to the device (GPU or CPU)
        y_train_full, X_train, X_val, X_test, y_train, y_val, y_test = (
            y_train_full.to(device),
            X_train.to(device),
            X_val.to(device),
            X_test.to(device),
            y_train.to(device),
            y_val.to(device),
            y_test.to(device),
        )

        num_classes = len(torch.unique(y))
        # print("num_classes: ", num_classes)
        time_record = []
        loss_records = []
        os.makedirs(f"optModel/{name}", exist_ok=True)
        if X_train.shape[1] > 100:
            depthValue = default_deep
        else:
            depthValue = default_shallow

        # set seed for model
        random_state = seed
        set_seed(random_state)
        model = DifferentiableDecisionTree(depthValue, num_classes, ranked_indices).to(
            device
        )
        # criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        start = time.time()
        epochs = 10000
        save_interval = 100

        patience = 10
        threshold = 0.0001
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_weights = None
        optimal_epoch = 0

        for epoch in range(1, epochs + 1):
            # model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val).item()
            if best_val_loss - val_loss > threshold:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!", flush=True)
                print(f"Depth {depthValue}, optimal Epoch {epoch}", flush=True)
                optimal_epoch = epoch
                break

        if best_model_weights:
            model.load_state_dict(best_model_weights)
            optimal_epoch = epoch
        train_time = time.time() - start

        # train_accuracy = evaluate_model(model, X_train, y_train)
        (
            train_accuracy,
            train_f1_macro,
            # train_roc_auc,
            train_accuracy_time,
        ) = evaluate_model(model, X_train, y_train, num_classes)

        (
            test_accuracy,
            test_f1_macro,
            # test_roc_auc,
            test_accuracy_time,
        ) = evaluate_model(model, X_test, y_test, num_classes)

        # Store results for this depth and seed
        trial_result = {
            "Trial": trial + 1,
            "Seed": seed,
            "name": name,
            "Depth": depthValue,
            "Optimal Epoch": optimal_epoch,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Train F1 Macro": train_f1_macro,
            "Test F1 Macro": test_f1_macro,
            # "Train ROC AUC": train_roc_auc,
            # "Test ROC AUC": test_roc_auc,
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Train Time": train_time,
            "Test Accuracy Time": test_accuracy_time,
        }
        all_trials_results.append(trial_result)
        results_df = pd.DataFrame(all_trials_results)
        results_df.to_csv(f"optModel/{name}/{name}_all_trials_results.csv", index=False)

        print(
            f"trial: {trial+1},seed:{seed},name: {name}, depth: {depthValue}, optimal epoch: {optimal_epoch}, train accuracy:{train_accuracy}, test accuracy:{test_accuracy},train_f1_macro:{train_f1_macro},test_f1_macro:{test_f1_macro},train time: {train_time}, test accuracy time: {test_accuracy_time}",
            flush=True,
        )

    statistics = []
    # Identify all unique depths in the trials
    unique_depths = set(trial["Depth"] for trial in all_trials_results)

    # Loop through each unique depth
    for depth in unique_depths:
        # Filter the trials to get only those with the current depth
        filtered_trials = [
            trial for trial in all_trials_results if trial["Depth"] == depth
        ]

        # Extract Test Accuracy values for the filtered trials
        test_accuracy_results = [trial["Test Accuracy"] for trial in filtered_trials]
        test_f1_results = [trial["Test F1 Macro"] for trial in filtered_trials]
        # test_roc_results = [trial["Test ROC AUC"] for trial in filtered_trials]
        training_time = [trial["Train Time"] for trial in filtered_trials]
        test_acc_time = [trial["Test Accuracy Time"] for trial in filtered_trials]
        mean_test_accuracy = np.mean(test_accuracy_results)
        std_test_accuracy = np.std(test_accuracy_results)
        mean_f1_score = np.mean(test_f1_results)
        std_f1_score = np.std(test_f1_results)
        # mean_roc_auc = np.mean(test_roc_results)
        # std_roc_auc = np.std(test_roc_results)
        mean_training_time = np.mean(training_time)
        mean_test_acc_time = np.mean(test_acc_time)

        dataset_name = filtered_trials[0]["name"]
        statistics.append(
            {
                "Dataset Name": dataset_name,
                "Depth": depth,
                "Mean Test Accuracy": mean_test_accuracy,
                "Std Test Accuracy": std_test_accuracy,
                "Mean F1 Score": mean_f1_score,
                "Std F1 Score": std_f1_score,
                # "Mean ROC AUC": mean_roc_auc,
                # "Std ROC AUC": std_roc_auc,
                "Mean Training Time": mean_training_time,
                "Mean Test Acc Time": mean_test_acc_time,
            }
        )

        statistics_df = pd.DataFrame(statistics)
        statistics_df.to_csv(
            f"optModel/{name}/Statistics_summary_{dataset_name}.csv",
            index=False,
        )

        # Print the mean and standard deviation for the current depth
        print(f"Mean Test Accuracy for Depth {depth}: {mean_test_accuracy:.4f}")
        print(
            f"Standard Deviation of Test Accuracy for Depth {depth}: {std_test_accuracy:.4f}"
        )
        print(
            f"All trials' results are saved in: optModel/{name}/{name}_all_trials_results.csv"
        )
        print(
            f"Statistic summary is saved in: optModel/{name}/Statistics_summary_{dataset_name}.csv"
        )

    # idx = int(sys.argv[1])
