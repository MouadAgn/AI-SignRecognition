from __future__ import annotations
import argparse, csv, os
from pathlib import Path
import numpy as np
import cv2, mediapipe as mp
from .landmark_features import build_feature_vector

# Par défaut: lettres statiques A..Y sans J/Z (tu peux en ajouter via --keep)
DEFAULT_KEEP = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # 24 classes

def iter_class_folders(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        files = [f for f in filenames if f.lower().endswith((".jpg",".jpeg",".png"))]
        if files:
            label = Path(dirpath).name
            yield Path(dirpath), label, sorted(files)

def main(args):
    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Classes à garder
    keep = set([c.lower() for c in (args.keep or DEFAULT_KEEP)])
    max_per = int(args.max_per_class)

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1,
        model_complexity=1, min_detection_confidence=0.5
    )

    kept_total = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label"] + [f"f{i}" for i in range(102)])

        for folder, label, files in iter_class_folders(root):
            lab_norm = label.lower()
            # Le dataset a des dossiers 'A','B',... et aussi 'space','nothing','del'
            if lab_norm in ("space","nothing","del"):
                # ignorer par défaut; ajoute-les avec --keep si tu veux
                pass
            elif len(label) == 1:
                lab_norm = label.lower()
            else:
                continue

            if lab_norm not in keep:
                continue

            n_ok = 0
            for fn in files:
                if n_ok >= max_per: break
                p = folder / fn
                img = cv2.imread(str(p))
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(img_rgb)
                if not res.multi_hand_landmarks:
                    continue
                lms = res.multi_hand_landmarks[0].landmark
                lms_xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
                handed = None
                try:
                    handed = res.multi_handedness[0].classification[0].label
                except Exception:
                    pass
                feats = build_feature_vector(lms_xy, handed)
                w.writerow([label.upper()] + feats.tolist())
                n_ok += 1

            kept_total += n_ok
            print(f"{label}: {n_ok} échantillons retenus")

    hands.close()
    print(f"Terminé → {out_csv} ({kept_total} lignes)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/asl_alphabet/asl_alphabet_train/asl_alphabet_train")
    ap.add_argument("--out_csv", type=str, default="data/landmarks/features.csv")
    ap.add_argument("--max_per_class", type=int, default=400)
    ap.add_argument("--keep", nargs="*", default=None, help="Ex: A B C ... space nothing del")
    args = ap.parse_args()
    main(args)
 