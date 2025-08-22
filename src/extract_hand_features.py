# src/extract_hand_features.py
# Modes:
#   - collect : capture webcam -> extrait des features (landmarks Mediapipe) -> append CSV
#   - train   : lit CSV -> split -> StandardScaler + MLPClassifier (early_stopping + class_weight balanced)
#               -> rapport, confusion matrix, PR-curves -> sauvegarde modèle
#   - demo    : charge le modèle -> prédiction en temps réel -> affiche label + construit une phrase simple
#
# Lancer:
#   uv run python src/extract_hand_features.py collect --label Open --seconds 20
#   uv run python src/extract_hand_features.py train
#   uv run python src/extract_hand_features.py demo
#
# Dépendances: mediapipe==0.10.21, opencv-python, scikit-learn, numpy, matplotlib, tqdm, rich
# Sorties: data/gestures_features.csv, outputs/models/gesture_mlp.joblib, outputs/figures/*

from __future__ import annotations
import argparse, csv, math, time, json
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import joblib
from rich import print

from utils_metrics import plot_confusion_matrix, plot_pr_curve, save_jsonl

# ---------- Constantes & mapping "geste -> mot" (proto traduction) ----------
DEFAULT_LABELS = ["Open", "Fist", "Pinch", "Point", "ThumbsUp"]

# Petit lexique FR (démonstration) : à adapter selon votre jeu
LEXICON_FR = {
    "Open": "bonjour",
    "Fist": "stop",
    "Pinch": "sélection",
    "Point": "là-bas",
    "ThumbsUp": "oui"
}

# Landmarks index
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP   = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index":  (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring":   (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky":  (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

THRESH = {
    "extended_margin": 0.035,
    "pinch_dist": 0.06,
    "thumb_up_y": 0.04,
}

# ---------- Utils visu ----------
def put_text(img, text, org, scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

def draw_landmarks(image, hand_landmarks):
    h, w = image.shape[:2]
    idx_pairs = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]
    pts = []
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x,y))
        cv2.circle(image, (x,y), 2, (0,255,0), -1, cv2.LINE_AA)
    for i,j in idx_pairs:
        cv2.line(image, pts[i], pts[j], (0,200,255), 1, cv2.LINE_AA)

# ---------- Features ----------
def norm_dist_xy(a, b):
    dx, dy = a.x - b.x, a.y - b.y
    return math.hypot(dx, dy)

def finger_extended(lm, wrist, mcp, pip, dip, tip):
    d_tip = norm_dist_xy(lm[tip], wrist)
    d_pip = norm_dist_xy(lm[pip], wrist)
    d_dip = norm_dist_xy(lm[dip], wrist)
    return float((d_tip > d_pip + THRESH["extended_margin"]) and (d_tip > d_dip + THRESH["extended_margin"]))

def angle_between(a, b):
    # angle atan2 du vecteur a->b
    vx, vy = (b.x - a.x), (b.y - a.y)
    return math.atan2(vy, vx)  # radians

def extract_features_from_landmarks(lm) -> list[float]:
    """Renvoie un vecteur de features robustes & compacts, normalisés par la taille de la main."""
    wrist = lm[WRIST]
    # Echelle = diagonale de la bbox des landmarks (évite la dépendance à la distance caméra)
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    scale = math.hypot((max(xs)-min(xs)), (max(ys)-min(ys))) + 1e-6

    # Distances (tip -> wrist) normalisées
    d_idx = norm_dist_xy(lm[INDEX_TIP], wrist)   / scale
    d_mid = norm_dist_xy(lm[MIDDLE_TIP], wrist)  / scale
    d_rng = norm_dist_xy(lm[RING_TIP], wrist)    / scale
    d_pky = norm_dist_xy(lm[PINKY_TIP], wrist)   / scale
    d_thb = norm_dist_xy(lm[THUMB_TIP], wrist)   / scale

    # États "étendu" (0/1) pour 4 doigts (sans le pouce) + pouce étendu
    ext_idx = finger_extended(lm, wrist, *FINGERS["index"])
    ext_mid = finger_extended(lm, wrist, *FINGERS["middle"])
    ext_rng = finger_extended(lm, wrist, *FINGERS["ring"])
    ext_pky = finger_extended(lm, wrist, *FINGERS["pinky"])
    # pouce étendu (comparaison TIP vs IP/MCP)
    thumb_d_tip = norm_dist_xy(lm[THUMB_TIP], wrist)
    thumb_d_ip  = norm_dist_xy(lm[THUMB_IP], wrist)
    thumb_d_mcp = norm_dist_xy(lm[THUMB_MCP], wrist)
    ext_thb = float((thumb_d_tip > thumb_d_ip + THRESH["extended_margin"]) and (thumb_d_tip > thumb_d_mcp + THRESH["extended_margin"]))

    # Pinch distance (thumb tip - index tip) normalisée
    pinch = norm_dist_xy(lm[THUMB_TIP], lm[INDEX_TIP]) / scale

    # Angles des axes index & pouce (orientation)
    ang_index = angle_between(lm[INDEX_MCP], lm[INDEX_TIP])  # radians [-pi, pi]
    ang_thumb = angle_between(lm[THUMB_MCP], lm[THUMB_TIP])
    # sin/cos pour enlever la discontinuité d'angle
    sin_ci, cos_ci = math.sin(ang_index), math.cos(ang_index)
    sin_ct, cos_ct = math.sin(ang_thumb), math.cos(ang_thumb)

    feats = [
        d_idx, d_mid, d_rng, d_pky, d_thb,
        ext_idx, ext_mid, ext_rng, ext_pky, ext_thb,
        pinch,
        sin_ci, cos_ci, sin_ct, cos_ct
    ]
    return feats

# ---------- CSV I/O ----------
def ensure_data_csv(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        header = [f"f{i}" for i in range(15)] + ["label"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

def append_sample(csv_path: Path, feats: list[float], label: str):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([*map(lambda x: f"{x:.6f}", feats), label])

# ---------- Modes ----------
def run_collect(args):
    csv_path = Path("data/gestures_features.csv")
    ensure_data_csv(csv_path)

    label = args.label
    assert label in DEFAULT_LABELS, f"Label inconnu: {label} (choisis parmi {DEFAULT_LABELS})"

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam (index 0).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    print(f"[bold green]Collecte[/bold green] label={label} pendant ~{args.seconds}s. Touche [Q] pour arrêter plus tôt.")
    start = time.time(); saved = 0; mirror = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if mirror: frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                feats = extract_features_from_landmarks(lm)
                append_sample(csv_path, feats, label)
                saved += 1
                put_text(frame, f"Sauvegardes: {saved}", (10, 30))
                draw_landmarks(frame, result.multi_hand_landmarks[0])

            put_text(frame, f"Label: {label}  |  [Q] Quitter", (10, frame.shape[0]-10), scale=0.6, thickness=1)
            cv2.imshow("Collecte features", frame)

            if (time.time() - start) >= args.seconds:
                break
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break
    finally:
        cap.release(); cv2.destroyAllWindows(); hands.close()

    print(f"[bold]Terminé.[/bold] Echantillons ajoutés: {saved} -> {csv_path}")

def run_train(args):
    csv_path = Path("data/gestures_features.csv")
    assert csv_path.exists(), f"CSV introuvable: {csv_path}. Lance d'abord 'collect'."

    # Charger CSV
    rows = list(csv.DictReader(open(csv_path, "r", encoding="utf-8")))
    X = np.array([[float(r[f"f{i}"]) for i in range(15)] for r in rows], dtype=np.float32)
    y_lbl = np.array([r["label"] for r in rows], dtype=object)

    # Encodage labels -> y
    le = LabelEncoder()
    y = le.fit_transform(y_lbl)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    print(f"[bold]Classes:[/bold] {class_names} (n={num_classes})  |  Samples={len(X)}")

    # Split stratifié
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)

    # Pipeline: StandardScaler + MLPClassifier (early_stopping)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=False
        ))
    ])

    print("[bold green]Entraînement MLP (early_stopping, class_weight='balanced')...[/bold green]")
    clf.fit(X_tr, y_tr)

    # Eval
    te_prob = clf.predict_proba(X_te)
    te_pred = te_prob.argmax(axis=1)

    report = classification_report(y_te, te_pred, target_names=class_names, digits=4)
    print("\n[bold cyan]Classification report (test):[/bold cyan]\n" + report)

    # Figures
    plot_confusion_matrix(y_te, te_pred, class_names, "outputs/figures/gestures_confusion.png")
    # PR-AUC (macro)
    y_onehot = np.zeros((len(y_te), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_te)), y_te] = 1.0
    macro_ap = plot_pr_curve(y_onehot, te_prob, class_names, "outputs/figures/gestures_pr.png")
    print(f"[bold magenta]Macro PR-AUC (test): {macro_ap:.4f}[/bold magenta]")

    # Sauvegarde modèle + labels
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, "outputs/models/gesture_mlp.joblib")
    json.dump(class_names, open("outputs/models/gesture_labels.json", "w", encoding="utf-8"))
    save_jsonl("outputs/logs/gestures_run.jsonl", {
        "samples": len(X),
        "classes": class_names,
        "test_size": args.test_size,
        "macro_pr_auc": float(macro_ap)
    })
    print("[bold green]Modèle sauvegardé → outputs/models/gesture_mlp.joblib[/bold green]")

def run_demo(args):
    # Charger modèle + labels
    model_path = Path("outputs/models/gesture_mlp.joblib")
    labels_path = Path("outputs/models/gesture_labels.json")
    assert model_path.exists() and labels_path.exists(), "Lancez d'abord 'train'."
    clf = joblib.load(model_path)
    class_names = json.load(open(labels_path, "r", encoding="utf-8"))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): raise RuntimeError("Impossible d'ouvrir la webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    label_buf = deque(maxlen=8)
    words = []  # phrase en construction
    mirror = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if mirror: frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            label = "NoHand"
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                feats = np.array(extract_features_from_landmarks(lm), dtype=np.float32)[None, :]
                prob = clf.predict_proba(feats)[0]
                idx = int(prob.argmax())
                label = class_names[idx]

                # dessiner
                draw_landmarks(frame, res.multi_hand_landmarks[0])
                put_text(frame, f"{label} ({prob[idx]:.2f})", (10, 40))

                label_buf.append(label)
                # si stable -> convertir en mot et l'ajouter (auto)
                if len(label_buf) == label_buf.maxlen:
                    maj = Counter(label_buf).most_common(1)[0][0]
                    # éviter d'ajouter la même chose en boucle
                    if not words or (LEXICON_FR.get(maj, maj) != words[-1]):
                        words.append(LEXICON_FR.get(maj, maj))
                    label_buf.clear()

            # Affichage phrase & aides
            put_text(frame, f"Phrase: {' '.join(words)[:60]}", (10, 70))
            put_text(frame, "Keys: [Q]uit [C]lear phrase [M]irror on/off", (10, frame.shape[0]-10), scale=0.6, thickness=1)

            cv2.imshow("Gestes -> Mots (MLP demo)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            elif key in (ord('c'), ord('C')): words.clear()
            elif key in (ord('m'), ord('M')): mirror = not mirror

    finally:
        cap.release(); cv2.destroyAllWindows(); hands.close()

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect", help="Collecter des features pour un label -> CSV")
    pc.add_argument("--label", type=str, choices=DEFAULT_LABELS, required=True)
    pc.add_argument("--seconds", type=int, default=20)

    pt = sub.add_parser("train", help="Entraîner MLP sur le CSV")
    pt.add_argument("--test_size", type=float, default=0.2)

    pd = sub.add_parser("demo", help="Démo temps réel avec le modèle entraîné")

    args = p.parse_args()
    if args.cmd == "collect":
        run_collect(args)
    elif args.cmd == "train":
        run_train(args)
    elif args.cmd == "demo":
        run_demo(args)

if __name__ == "__main__":
    main()