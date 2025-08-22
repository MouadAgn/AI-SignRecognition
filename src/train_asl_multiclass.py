# src/train_asl_multiclass.py
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
from tqdm import tqdm
from rich import print

from utils_metrics import (
    save_jsonl, plot_confusion_matrix, plot_pr_curve
)

# --------------------------------------------------------------------------------------
# Notes sur le dataset Kaggle "Sign Language MNIST":
# - Certaines versions mappent 24 classes (pas de J/Z car gestes dynamiques).
# - D'autres re-mappent sur 26 classes. Si vous avez 24 classes, adaptez CLASS_NAMES et
#   vérifiez les labels uniques dans les CSV pour ajuster automatiquement.
# --------------------------------------------------------------------------------------

def infer_class_names(labels: np.ndarray) -> list[str]:
    uniq = sorted(set(int(x) for x in labels))
    n = len(uniq)
    # Par défaut: lettres A..Z. Si 24 classes, on conserve A..Y sans J/Z (classique Kaggle).
    base = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    if n == 24:
        # Version Kaggle classique (sans J et Z)
        # On retire J (9) et Z (25) de l'affichage pour rester fidèle à l'étiquette d'origine.
        return [c for i, c in enumerate(base) if i not in (9, 25)]
    elif n == 26:
        return base
    else:
        # Sinon, on nomme génériquement C0..C{n-1}
        return [f"C{i}" for i in range(n)]

class ASLMNIST(Dataset):
    """Charge un CSV Kaggle: label,pixel1,...,pixel784  (28x28 gris)."""
    def __init__(self, csv_path: Path):
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            _ = next(reader)  # header
            for r in reader:
                rows.append(r)
        self.labels = np.array([int(r[0]) for r in rows], dtype=np.int64)
        pixels = np.array([r[1:] for r in rows], dtype=np.float32).reshape(-1, 28, 28)
        self.images = pixels / 255.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.images[idx][None, ...]  # (1,28,28)
        y = self.labels[idx]
        return torch.from_numpy(x), int(y)

class SmallCNN(nn.Module):
    """CNN compacte et robuste pour 28x28 (CPU-friendly)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (counts + 1e-6)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y_oh = np.zeros((len(y), num_classes), dtype=np.float32)
    y_oh[np.arange(len(y)), y] = 1.0
    return y_oh

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, corr, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        corr += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, corr / n

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes: int):
    model.eval()
    loss_sum, corr, n = 0.0, 0, 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        corr += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
        ys.append(y.cpu().numpy())
        ps.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    acc = corr / n
    return loss_sum / n, acc, y_true, y_prob

def main(args):
    # Dossiers
    out_models = Path("outputs/models"); out_models.mkdir(parents=True, exist_ok=True)
    out_logs   = Path("outputs/logs");   out_logs.mkdir(parents=True, exist_ok=True)
    out_figs   = Path("outputs/figures");out_figs.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bold green]Device:[/bold green] {device}")

    # Données
    train_csv = Path(args.data_dir) / "sign_mnist_train.csv"
    test_csv  = Path(args.data_dir) / "sign_mnist_test.csv"
    train_ds = ASLMNIST(train_csv)
    test_ds  = ASLMNIST(test_csv)

    # Inférer les classes (24 vs 26) à partir du train
    CLASS_NAMES = infer_class_names(train_ds.labels)
    num_classes = len(CLASS_NAMES)
    print(f"[bold]Classes détectées:[/bold] {num_classes} → {CLASS_NAMES}")

    # Split train/val
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(train_ds))
    tr_idx, val_idx = train_test_split(
        idx, test_size=0.1, stratify=train_ds.labels, random_state=42
    )
    subset = torch.utils.data.Subset
    train_split = subset(train_ds, tr_idx)
    val_split   = subset(train_ds, val_idx)

    # Sampler / weights
    criterion = nn.CrossEntropyLoss()
    sampler = None

    if args.use_class_weights.lower() == "true":
        class_w = compute_class_weights(train_ds.labels[tr_idx], num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w)

    if args.sampler == "weighted":
        lbls = train_ds.labels[tr_idx]
        class_counts = np.bincount(lbls, minlength=num_classes).astype(np.float32)
        weights = 1.0 / (class_counts[lbls] + 1e-6)
        sampler = WeightedRandomSampler(weights=torch.tensor(weights), num_samples=len(lbls), replacement=True)

    # Loaders
    train_loader = DataLoader(train_split, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_split, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Modèle/opt
    model = SmallCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    # Entraînement + early stopping
    best_val_loss = float("inf")
    wait = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_val, p_val = eval_epoch(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_loss)

        # Logs
        log = {
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss,   "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        }
        save_jsonl("outputs/logs/asl_run.jsonl", log)
        print(f"[bold]Epoch {epoch}[/bold] "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.3f}")

        # Early stopping (sauvegarde meilleur modèle)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), "outputs/models/asl_cnn.pt")
        else:
            wait += 1
            if wait >= args.early_patience:
                print("[yellow]Early stopping triggered[/yellow]")
                break

    # Évaluation test (meilleur checkpoint)
    model.load_state_dict(torch.load("outputs/models/asl_cnn.pt", map_location=device))
    test_loss, test_acc, y_test, p_test = eval_epoch(model, test_loader, criterion, device, num_classes)

    # Figures & rapports
    y_pred = p_test.argmax(1)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n[bold cyan]Classification report (test):[/bold cyan]\n" + report)

    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, "outputs/figures/confusion_matrix.png")

    y_test_oh = one_hot(y_test, num_classes)
    macro_ap = plot_pr_curve(y_test_oh, p_test, CLASS_NAMES, "outputs/figures/pr_curve.png")
    print(f"[bold magenta]Macro PR-AUC (test): {macro_ap:.4f}[/bold magenta]")

    # Résumé final
    save_jsonl("outputs/logs/asl_run.jsonl", {
        "final_test_loss": test_loss, "final_test_acc": test_acc, "macro_pr_auc": macro_ap
    })

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/asl_mnist")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early_patience", type=int, default=5)
    p.add_argument("--use_class_weights", type=str, default="true")  # "true"|"false"
    p.add_argument("--sampler", type=str, choices=["none", "weighted"], default="none")
    args = p.parse_args()
    main(args)
