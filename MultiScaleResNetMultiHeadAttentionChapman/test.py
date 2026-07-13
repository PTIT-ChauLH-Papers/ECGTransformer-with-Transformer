import torch
from Chapman import ChapmanTestDataset, ChapmanTrainDataset
from MultiScaleCNN import MultiScaleCNN
import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# -------------------------
# Config
# -------------------------
path = "../data/chapman/csv/raw.csv"

NUM_EPOCHS   = 60
CLASS_NAMES  = ["AFIB", "GSVT", "SB", "SR"]   # adjust if needed


# -------------------------
# Reproducibility
# -------------------------
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
# Train / Evaluate
# -------------------------
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device).long()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return running_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        outputs = model(inputs)
        running_loss += loss_fn(outputs, labels).item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    return running_loss / max(1, len(loader)), acc, all_preds, all_labels


# -------------------------
# Metrics from CM
# -------------------------
def compute_metrics(all_labels, all_preds, class_names):
    cm      = confusion_matrix(all_labels, all_preds)
    n       = cm.shape[0]
    support = cm.sum(axis=1)              # samples per class

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # Per-class
    prec_per = np.where(TP+FP > 0, TP/(TP+FP), 0.0)
    rec_per  = np.where(TP+FN > 0, TP/(TP+FN), 0.0)
    spec_per = np.where(TN+FP > 0, TN/(TN+FP), 0.0)
    f1_per   = np.where(prec_per+rec_per > 0,
                        2*prec_per*rec_per/(prec_per+rec_per), 0.0)
    acc_per  = np.where(TP+FP+FN+TN > 0,
                        (TP+TN)/(TP+FP+FN+TN), 0.0)

    # Macro avg
    def macro(x): return x.mean()

    # Weighted avg
    w = support / support.sum()
    def weighted(x): return (x * w).sum()

    # Micro avg
    TP_s, FP_s, FN_s, TN_s = TP.sum(), FP.sum(), FN.sum(), TN.sum()
    prec_micro = TP_s/(TP_s+FP_s) if TP_s+FP_s > 0 else 0.0
    rec_micro  = TP_s/(TP_s+FN_s) if TP_s+FN_s > 0 else 0.0
    spec_micro = TN_s/(TN_s+FP_s) if TN_s+FP_s > 0 else 0.0
    f1_micro   = (2*prec_micro*rec_micro/(prec_micro+rec_micro)
                  if prec_micro+rec_micro > 0 else 0.0)
    acc_overall = TP_s / cm.sum()   # same for micro

    # Build summary rows
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "Class":       name,
            "Accuracy":    f"{acc_per[i]*100:.2f}%",
            "Precision":   f"{prec_per[i]:.4f}",
            "Recall":      f"{rec_per[i]:.4f}",
            "Specificity": f"{spec_per[i]:.4f}",
            "F1 Score":    f"{f1_per[i]:.4f}",
            "Support":     int(support[i]),
        })

    rows.append({   # separator row
        "Class": "─" * 12,
        "Accuracy": "─"*9, "Precision": "─"*9, "Recall": "─"*9,
        "Specificity": "─"*9, "F1 Score": "─"*9, "Support": "─"*7,
    })
    rows.append({
        "Class":       "Macro avg",
        "Accuracy":    f"",
        "Precision":   f"{macro(prec_per):.4f}",
        "Recall":      f"{macro(rec_per):.4f}",
        "Specificity": f"{macro(spec_per):.4f}",
        "F1 Score":    f"{macro(f1_per):.4f}",
        "Support":     int(support.sum()),
    })
    rows.append({
        "Class":       "Micro avg",
        "Accuracy":    f"",
        "Precision":   f"{prec_micro:.4f}",
        "Recall":      f"{rec_micro:.4f}",
        "Specificity": f"{spec_micro:.4f}",
        "F1 Score":    f"{f1_micro:.4f}",
        "Support":     int(support.sum()),
    })
    rows.append({
        "Class":       "Weighted avg",
        "Accuracy":    f"",
        "Precision":   f"{weighted(prec_per):.4f}",
        "Recall":      f"{weighted(rec_per):.4f}",
        "Specificity": f"{weighted(spec_per):.4f}",
        "F1 Score":    f"{weighted(f1_per):.4f}",
        "Support":     int(support.sum()),
    })

    df_metrics = pd.DataFrame(rows)

    # Print to console
    print("\n" + "=" * 75)
    print("                       EVALUATION METRICS TABLE")
    print("=" * 75)
    print(df_metrics.to_string(index=False))
    print("=" * 75)

    return dict(
        confusion_matrix = cm,
        df_metrics       = df_metrics,
        acc              = acc_overall,
        prec_macro       = macro(prec_per),
        rec_macro        = macro(rec_per),
        spec_macro       = macro(spec_per),
        f1_macro         = macro(f1_per),
        prec_micro       = prec_micro,
        rec_micro        = rec_micro,
        spec_micro       = spec_micro,
        f1_micro         = f1_micro,
        prec_weighted    = weighted(prec_per),
        rec_weighted     = weighted(rec_per),
        spec_weighted    = weighted(spec_per),
        f1_weighted      = weighted(f1_per),
    )


# -------------------------
# Plot: Confusion Matrix + Metrics Table
# -------------------------
def plot_results(cm, df_metrics, class_names,
                cm_path="confusion_matrix.png",
                table_path="metrics_table.png"):

    # ── 1. Confusion matrix (counts + normalised) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalised)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    # ── 2. Metrics table image ──────────────────────────────────────────────
    # Remove separator row before plotting
    df_plot = df_metrics[~df_metrics["Class"].str.startswith("─")].copy()

    col_labels = list(df_plot.columns)
    cell_text  = df_plot.values.tolist()
    n_rows     = len(cell_text)

    # Row colours: per-class = light blue, summary rows = light orange
    n_classes   = len(class_names)
    row_colors  = [["#dce8f7"] * len(col_labels)] * n_classes
    row_colors += [["#fde8c8"] * len(col_labels)] * (n_rows - n_classes)

    fig, ax = plt.subplots(figsize=(13, 0.55 * n_rows + 1.5))
    ax.axis("off")

    tbl = ax.table(
        cellText   = cell_text,
        colLabels  = col_labels,
        cellLoc    = "center",
        loc        = "center",
        cellColours= row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    # Style header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#1f4e79")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Metrics Summary Table", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics table saved  → {table_path}")


# -------------------------
# Main
# -------------------------
PARAMS = {
    "lr": 0.0001,
    "batch_size": 32,
    "weight_decay": 0.0001,
    "dropout_rate": 0.3,
    "kernel_size": 7,
    "n_heads": 16,
    "output_dim": 512,
    "attention_num_layers": 4,
}

if __name__ == "__main__":
    data = pd.read_csv(path, header=None, skiprows=0)
    train_data_pd, test_data_pd = train_test_split(data, test_size=0.2, random_state=42)

    data_train = ChapmanTrainDataset(train_data_pd)
    data_test  = ChapmanTestDataset(test_data_pd)

    batch_size   = PARAMS["batch_size"]
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True,  drop_last=False, pin_memory=True, num_workers=4)
    test_loader  = DataLoader(data_test,  batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)

    model = MultiScaleCNN(
        input_channels=1,
        kernel_size=PARAMS["kernel_size"],
        dropout_rate=PARAMS["dropout_rate"],
        n_heads=PARAMS["n_heads"],
        output_dim=PARAMS["output_dim"],
        attention_num_layers=PARAMS["attention_num_layers"],
    ).to(device)
    
    # First run speed up with torch.compile (if available) - can be ~2-3x faster on GPU for this model
    if (hasattr(model, "compile")):
        model = torch.compile(model)
        
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PARAMS["lr"],
        weight_decay=PARAMS["weight_decay"],
        betas=(0.9, 0.99),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | loss={tr_loss:.4f} | acc={tr_acc*100:.2f}%")

    # Evaluation
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, loss_fn)
    print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc*100:.2f}%")

    # Metrics
    m = compute_metrics(all_labels, all_preds, class_names=CLASS_NAMES)

    # Plots
    plot_results(
        m["confusion_matrix"], m["df_metrics"], CLASS_NAMES,
        cm_path="confusion_matrix.png",
        table_path="metrics_table.png",
    )

    # Save CSV
    summary_rows = {
        "lr":              [PARAMS["lr"]],
        "batch_size":      [PARAMS["batch_size"]],
        "weight_decay":    [PARAMS["weight_decay"]],
        "kernel_size":     [PARAMS["kernel_size"]],
        "dropout_rate":    [PARAMS["dropout_rate"]],
        "nheads":          [PARAMS["n_heads"]],
        "attention_num_layers": [PARAMS["attention_num_layers"]],
    }
    pd.DataFrame(summary_rows).to_csv("results_fixed.csv", index=False)
    m["df_metrics"].to_csv("metrics_table.csv", index=False)
    print("Results saved → results_fixed.csv")
    print("Metrics table saved → metrics_table.csv")