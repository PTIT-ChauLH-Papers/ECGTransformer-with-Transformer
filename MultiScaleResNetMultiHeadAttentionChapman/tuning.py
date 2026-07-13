import torch
from Chapman import ChapmanTestDataset, ChapmanTrainDataset
from pathlib import Path
from MultiScaleCNN import MultiScaleCNN
import random
import numpy as np
import os
import pandas as pd
import optuna
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


path = "../data/chapman/csv/raw.csv"
data = pd.read_csv(path, header=None, skiprows=0)
train_data_pd, test_data_pd = train_test_split(data, test_size=0.2, random_state=42)
data_train = ChapmanTrainDataset(train_data_pd)
data_test = ChapmanTestDataset(test_data_pd)

def set_seed(seed_value=42):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Utilities
# -------------------------
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Lists to store all predictions and labels for F1 calculation
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        # Calculate Accuracy components
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Collect data for Scikit-Learn
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Final Metrics
    avg_loss = running_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    
    # Micro-F1 Calculation
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    return avg_loss, accuracy, micro_f1

# -------------------------
# Split train -> train/val
# -------------------------
val_ratio = 0.2
n_total = len(data_train)
n_val = int(n_total * val_ratio)
n_tr = n_total - n_val

generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(data_train, [n_tr, n_val], generator=generator)

loss_fn = torch.nn.CrossEntropyLoss()


# -------------------------
# Optuna objective: tune lr & batch_size
# -------------------------
def objective(trial: optuna.Trial):
    try:
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)    
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)    
        batch_size = trial.suggest_categorical("batch_size", [16,32])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)    
        kernel_size = trial.suggest_categorical("kernel_size", [5, 6, 7, 8, 9])
        n_heads = trial.suggest_categorical("n_heads", [4,8,16,32])
        output_dim = trial.suggest_categorical("output_dim", [128, 256, 512, 1024])
        attention_num_layers = trial.suggest_categorical("attention_num_layers", [1, 2, 3, 4, 5])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

        model = MultiScaleCNN(
            input_channels=1,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            n_heads=n_heads,
            output_dim=output_dim,
            attention_num_layers=attention_num_layers
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.99)
        )

        max_epochs = 20 
        best_val_acc = 0.0

        for epoch in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer, loss_fn)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn)

            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            best_val_acc = max(best_val_acc, val_acc)

        return best_val_acc
    except torch.cuda.OutOfMemoryError:
        # If a trial hits OOM, tell Optuna to skip it and keep going
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned() 
    finally:
        # 2. CRITICAL: Clean up every single time
        import gc
        del model
        gc.collect()
        torch.cuda.empty_cache()

def run_study(n_trials=25):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("Best validation micro-F1:", study.best_value)
    print("Best params:", study.best_params)

    return study


# -------------------------
# Final train with best params (60 epochs) + test
# -------------------------
def train_final_and_test(best_params):
    print(best_params)
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]

    full_train_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    model = MultiScaleCNN(
        input_channels=1,
        kernel_size=best_params["kernel_size"],
        dropout_rate=best_params["dropout_rate"],
        n_heads=best_params["n_heads"],
        output_dim=best_params["output_dim"],
        attention_num_layers=best_params["attention_num_layers"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=best_params["weight_decay"],
        betas=(0.9, 0.99)
    )

    num_epochs = 60

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(
            model,
            full_train_loader,
            optimizer,
            loss_fn
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={tr_loss:.4f} | "
            f"train_acc={tr_acc * 100:.2f}%"
        )

    test_loss, test_acc, test_f1_micro = evaluate(
        model,
        test_loader,
        loss_fn
    )

    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Micro-F1: {test_f1_micro * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    df = pd.DataFrame({
        "lr": [lr],
        "batch_size": [batch_size],
        "weight_decay": [best_params["weight_decay"]],
        "kernel_size": [best_params["kernel_size"]],
        "dropout_rate": [best_params["dropout_rate"]],
        "n_heads": [best_params["n_heads"]],
        "output_dim": [best_params["output_dim"]],
        "attention_num_layers": [best_params["attention_num_layers"]],
    })

    df.to_csv("results_optuna.csv", index=False)

    return model


if __name__ == "__main__":
    study = run_study(n_trials=50)
    best_params = study.best_params
    train_final_and_test(best_params)