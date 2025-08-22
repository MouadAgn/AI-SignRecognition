from __future__ import annotations
import argparse, time, json
from collections import deque, Counter
from pathlib import Path
import numpy as np
import joblib
import cv2, mediapipe as mp
from .landmark_features import build_feature_vector

MODEL_PATH = Path("outputs/models/landmarks_mlp.joblib")
LABELS_JSON = Path("outputs/models/landmark_labels.json")

def put_text(img, text, org, s=0.7, th=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), th+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255,255,255), th, cv2.LINE_AA)

def main(args):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modèle landmarks introuvable. Entraîne d'abord (train_landmarks_mlp.py).")
    classes = json.load(open(LABELS_JSON,"r",encoding="utf-8"))
    clf = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): raise RuntimeError("Webcam introuvable")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        model_complexity=1, min_detection_confidence=args.det_conf, min_tracking_confidence=args.trk_conf
    )

    mirror=True
    ring = deque(maxlen=args.stable_frames)
    last_letter_time = 0.0
    word, phrase = [], []
    nohand_ms = 0.0
    prev_t = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if mirror: frame = cv2.flip(frame,1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t = time.monotonic(); dt = t - prev_t; prev_t = t

            res = hands.process(rgb)
            pred_txt = "-"; p_txt = 0.0
            if res.multi_hand_landmarks:
                nohand_ms = 0.0
                lms = res.multi_hand_landmarks[0].landmark
                lms_xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
                handed = None
                try:
                    handed = res.multi_handedness[0].classification[0].label
                except Exception:
                    pass

                feats = build_feature_vector(lms_xy, handed).reshape(1,-1)
                prob = clf.predict_proba(feats)[0]
                idx = int(prob.argmax()); p = float(prob[idx])
                pred = classes[idx]
                pred_txt, p_txt = pred, p

                if p >= args.min_prob:
                    ring.append(pred)

                # Dessin landmarks
                xs = (lms_xy[:,0]*w).astype(int); ys = (lms_xy[:,1]*h).astype(int)
                for i in range(21):
                    cv2.circle(frame, (xs[i], ys[i]), 2, (0,255,255), -1, cv2.LINE_AA)
            else:
                nohand_ms += dt*1000

            # Stabilisation + cooldown (lettre)
            now_ms = time.monotonic()*1000
            if len(ring)==ring.maxlen and (now_ms - last_letter_time) >= args.letter_cooldown_ms:
                maj = Counter(ring).most_common(1)[0][0]
                if not word or word[-1] != maj:
                    word.append(maj); last_letter_time = now_ms
                ring.clear()

            # Fin de mot par pause sans main
            if nohand_ms >= args.word_pause_ms and word:
                phrase.append("".join(word)); word.clear(); nohand_ms=0.0

            # HUD
            fps = 1.0/max(1e-6, dt)
            put_text(frame, f"FPS {fps:.1f}", (10,20), 0.6, 1)
            put_text(frame, f"Pred {pred_txt} ({p_txt:.2f})", (10,45))
            put_text(frame, f"Word: {''.join(word)}", (10,75))
            put_text(frame, f"Phrase: {' '.join([w for w in phrase if w])[:70]}", (10,105))
            put_text(frame, "Q quit | M mirror | BKSP del | ENTER commit word | SPACE space", (10,h-10), 0.5, 1)

            cv2.imshow("Landmarks MLP — Letters -> Word -> Phrase", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'),27): break
            elif k in (ord('m'),ord('M')): mirror = not mirror
            elif k==8 and word: word.pop()
            elif k==13 and word: phrase.append("".join(word)); word.clear()
            elif k==32: phrase.append("")

    finally:
        cap.release(); cv2.destroyAllWindows(); hands.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_conf", type=float, default=0.6)
    ap.add_argument("--trk_conf", type=float, default=0.6)
    ap.add_argument("--min_prob", type=float, default=0.80)
    ap.add_argument("--stable_frames", type=int, default=10)
    ap.add_argument("--letter_cooldown_ms", type=int, default=1000)
    ap.add_argument("--word_pause_ms", type=int, default=1500)
    args = ap.parse_args()
    main(args)
 