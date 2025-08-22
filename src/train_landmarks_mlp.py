from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
import joblib

IN_CSV = Path("data/landmarks/features.csv")
MODEL_PATH = Path("outputs/models/landmarks_mlp.joblib")
LABELS_JSON = Path("outputs/models/landmark_labels.json")
FIG_DIR = Path("outputs/figures")

def plot_confusion(cm, classes, outpath):
    plt.figure(figsize=(8,6))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    except Exception:
        plt.imshow(cm, cmap="Blues")
        plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=90)
        plt.yticks(ticks=np.arange(len(classes)), labels=classes)
    plt.title("Confusion matrix")
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def plot_pr_curves(y_true, proba, classes, outpath):
    plt.figure(figsize=(7,6))
    aps = []
    for i, c in enumerate(classes):
        y = (y_true == i).astype(int)
        p = proba[:, i]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        aps.append(ap)
        plt.plot(rec, prec, label=f"{c} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curves (macro AP={np.mean(aps):.3f})")
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
    return float(np.mean(aps))

def main(args):
    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} introuvable. Lance d'abord l'extraction de landmarks.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    X = df[[c for c in df.columns if c.startswith("f")]].values.astype(np.float32)
    y_text = df["label"].values
    le = LabelEncoder().fit(y_text)
    y = le.transform(y_text)
    classes = list(le.classes_)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128,64),
                              activation="relu",
                              alpha=1e-4,
                              batch_size=128,
                              learning_rate_init=1e-3,
                              max_iter=200,
                              early_stopping=True,
                              n_iter_no_change=10,
                              validation_fraction=0.2,
                              random_state=42,
                              verbose=False))
    ])
    pipe.fit(X_tr, y_tr)

    yv = pipe.predict(X_val); pv = pipe.predict_proba(X_val)
    yt = pipe.predict(X_te);  pt = pipe.predict_proba(X_te)
    print("Val acc:", accuracy_score(y_val, yv))
    print("Test acc:", accuracy_score(y_te, yt))
    print("\nClassification report (test):")
    print(classification_report(y_te, yt, target_names=classes, digits=4))

    cm = confusion_matrix(y_te, yt)
    plot_confusion(cm, classes, FIG_DIR/"landmarks_cm.png")
    map_ap = plot_pr_curves(y_te, pt, classes, FIG_DIR/"landmarks_pr.png")
    print("Macro PR-AUC (test):", map_ap)

    joblib.dump(pipe, MODEL_PATH)
    json.dump(classes, open(LABELS_JSON, "w", encoding="utf-8"))
    print(f"Saved: {MODEL_PATH}, {LABELS_JSON}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    main(args)
 