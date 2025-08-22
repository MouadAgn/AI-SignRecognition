from __future__ import annotations
import argparse, time, json, re
from collections import deque, Counter
from pathlib import Path
import numpy as np
import joblib
import cv2, mediapipe as mp

from .landmark_features import build_feature_vector

# --- Lexique & correction ---
from wordfreq import top_n_list
from rapidfuzz import process, fuzz

MODEL_PATH = Path("outputs/models/landmarks_mlp.joblib")
LABELS_JSON = Path("outputs/models/landmark_labels.json")

PUNCT = {".", ",", "!", "?", ":", ";"}

def put_text(img, text, org, s=0.7, th=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), th+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255,255,255), th, cv2.LINE_AA)

def build_vocab(lang: str, size: int = 50000) -> list[str]:
    # Liste de mots fréquents intégrée (offline). Lang "fr" ou "en".
    base = top_n_list(lang, size)  # <-- CORRIGÉ: n -> size (pas n_top)
    extras = ["asl","ai","bonjour","salut","merci","svp","ok"]
    s = set(base); s.update(extras)
    return list(s)    # Liste de mots fréquents intégrée (offline). Lang "fr" ou "en".
    base = top_n_list(lang, n_top=size)
    # Ajouts utiles pour démo (sigles, prénoms, etc.)
    extras = ["asl","ai","bonjour","salut","merci","svp","ok"]
    s = set(base); s.update(extras)
    return list(s)

def autocorrect(token: str, vocab: list[str], threshold: int = 85) -> str:
    """Retourne une suggestion proche si score RapidFuzz >= threshold, sinon le token tel quel."""
    if not token or len(token) < 2:
        return token
    low = token.lower()
    if low in vocab:  # déjà OK
        return token
    cand, score, _ = process.extractOne(low, vocab, scorer=fuzz.WRatio)
    if score >= threshold:
        # Respecte la casse: si token initial majuscule, capitalise la suggestion
        return cand.capitalize() if token[:1].isupper() else cand
    return token

def tidy_phrase(tokens: list[str]) -> str:
    """Jointure propre: espaces normaux, pas d'espace avant .,!?;:, capitalisation en début de phrase."""
    out = []
    prev = ""
    for t in tokens:
        if t in PUNCT:
            if out and out[-1] != " ":
                pass  # rien, on collera la ponctuation
            # colle la ponctuation au dernier mot
            if out and out[-1] == " ":
                out.pop()
            out.append(t)
            out.append(" ")
        else:
            if out and out[-1] not in (" ", "") and out[-1] not in PUNCT:
                out.append(" ")
            out.append(t)
    s = "".join(out).strip()
    # Capitaliser le début de phrase et après .!? (simple heuristique)
    s = re.sub(r"(^|\.\s+|\!\s+|\?\s+)(\w)", lambda m: m.group(1)+m.group(2).upper(), s)
    return s

def main(args):
    # Chargement modèle lettres (landmarks -> classe)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modèle landmarks introuvable. Entraîne d'abord src/train_landmarks_mlp.py.")
    classes = json.load(open(LABELS_JSON,"r",encoding="utf-8"))
    clf = joblib.load(MODEL_PATH)

    # Lexique (auto-corr)
    vocab = build_vocab(args.lang, size=args.vocab_size)

    # Webcam + MediaPipe Hands
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): raise RuntimeError("Webcam introuvable")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        model_complexity=1, min_detection_confidence=args.det_conf, min_tracking_confidence=args.trk_conf
    )

    # Buffers & tempo
    mirror=True
    ring = deque(maxlen=args.stable_frames)          # stabilisation lettre
    last_letter_time = 0.0
    word_chars: list[str] = []                       # lettres en cours
    phrase_tokens: list[str] = []                    # mots + ponctuation + espaces
    nohand_ms = 0.0
    prev_t = time.monotonic()

    # Aide: afficher un rappel des touches
    help_line = "Q quit | M mirror | BKSP del | ENTER commit word | SPACE space | . , ! ? ; : punct"

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

                # Dessine les landmarks
                xs=(lms_xy[:,0]*w).astype(int); ys=(lms_xy[:,1]*h).astype(int)
                for i in range(21):
                    cv2.circle(frame,(xs[i],ys[i]),2,(0,255,255),-1,cv2.LINE_AA)
            else:
                nohand_ms += dt*1000

            # Valider une lettre stabilisée (fenêtre + cooldown)
            now_ms = time.monotonic()*1000
            if len(ring)==ring.maxlen and (now_ms - last_letter_time) >= args.letter_cooldown_ms:
                maj = Counter(ring).most_common(1)[0][0]
                if not word_chars or word_chars[-1] != maj:
                    word_chars.append(maj); last_letter_time = now_ms
                ring.clear()

            # Fin de mot par pause sans main
            if nohand_ms >= args.word_pause_ms and word_chars:
                raw = "".join(word_chars)
                corrected = autocorrect(raw, vocab, threshold=args.autocorr_thresh)
                phrase_tokens.append(corrected)
                word_chars.clear()
                nohand_ms = 0.0

            # HUD
            fps = 1.0/max(1e-6, dt)
            put_text(frame, f"FPS {fps:.1f}", (10,20), 0.6, 1)
            put_text(frame, f"Letter {pred_txt} ({p_txt:.2f})", (10,45))
            put_text(frame, f"Word: {''.join(word_chars)}", (10,75))

            phrase_preview = tidy_phrase(phrase_tokens + (["".join(word_chars)] if word_chars else []))
            put_text(frame, f"Phrase: {phrase_preview[:70]}", (10,105), 0.6, 2)
            put_text(frame, help_line, (10,h-10), 0.5, 1)

            cv2.imshow("Letters → Words → Sentence (landmarks + MLP + autocorrect)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'),27): break
            elif k in (ord('m'),ord('M')): mirror = not mirror
            elif k==8 and word_chars:      # Backspace sur lettre
                word_chars.pop()
            elif k==13:                    # ENTER → commit mot immédiat
                if word_chars:
                    raw = "".join(word_chars)
                    corrected = autocorrect(raw, vocab, threshold=args.autocorr_thresh)
                    phrase_tokens.append(corrected)
                    word_chars.clear()
            elif k==32:                    # SPACE → espace (séparateur de mots)
                if word_chars:
                    raw = "".join(word_chars)
                    corrected = autocorrect(raw, vocab, threshold=args.autocorr_thresh)
                    phrase_tokens.append(corrected)
                    word_chars.clear()
                phrase_tokens.append("")    # espace explicite
            elif k in map(ord, [".",",","!","?",";",":"]):
                # Ponctuation rapide depuis le clavier
                ch = chr(k)
                if word_chars:
                    raw = "".join(word_chars)
                    corrected = autocorrect(raw, vocab, threshold=args.autocorr_thresh)
                    phrase_tokens.append(corrected)
                    word_chars.clear()
                phrase_tokens.append(ch)

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
    ap.add_argument("--lang", type=str, default="fr", choices=["fr","en"], help="Lexique pour l'autocorrection")
    ap.add_argument("--vocab_size", type=int, default=50000, help="Nb mots fréquents à charger")
    ap.add_argument("--autocorr_thresh", type=int, default=85, help="Seuil RapidFuzz (0-100)")
    args = ap.parse_args()
    main(args)
 