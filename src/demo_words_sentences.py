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

# ---------- Helpers visuels ----------
def put_text(img, text, org, s=0.7, th=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), th+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255,255,255), th, cv2.LINE_AA)

def convex_hull_mask(shape_wh, pts_xy, dilate_px=30, feather=21):
    """Construit un masque convexe (dilaté + bord adouci) autour de la main."""
    W, H = shape_wh
    mask = np.zeros((H, W), np.uint8)
    if not pts_xy:
        return mask, None
    hull = cv2.convexHull(np.array(pts_xy, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, 1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather|1, feather|1), 0)
    return mask, hull

def focus_effect(frame, mask, darken=0.2, inside_boost=1.05):
    """Assombrit le fond (1-darken) et rehausse légèrement l'intérieur du masque."""
    m = (mask.astype(np.float32)/255.0)[:, :, None]
    outside = (frame.astype(np.float32) * (1.0 - darken)).astype(np.uint8)
    inside  = np.clip(frame.astype(np.float32) * inside_boost, 0, 255).astype(np.uint8)
    return (inside * m + outside * (1.0 - m)).astype(np.uint8)

# ---------- Vocab / autocorrection ----------
def build_vocab(lang: str, size: int = 50000) -> list[str]:
    base = top_n_list(lang, size)  # <- correct (pas n_top)
    extras = ["asl","ai","bonjour","salut","merci","svp","ok"]
    s = set(base); s.update(extras)
    return list(s)

def autocorrect(token: str, vocab: list[str], threshold: int = 85) -> str:
    if not token or len(token) < 2:
        return token
    low = token.lower()
    if low in vocab:
        return token
    cand, score, _ = process.extractOne(low, vocab, scorer=fuzz.WRatio)
    if score >= threshold:
        return cand.capitalize() if token[:1].isupper() else cand
    return token

def tidy_phrase(tokens: list[str]) -> str:
    out = []
    for t in tokens:
        if t in PUNCT:
            if out and out[-1] == " ":
                out.pop()
            out.append(t); out.append(" ")
        else:
            if out and out[-1] not in (" ", "") and out[-1] not in PUNCT:
                out.append(" ")
            out.append(t)
    s = "".join(out).strip()
    s = re.sub(r"(^|\.\s+|\!\s+|\?\s+)(\w)", lambda m: m.group(1)+m.group(2).upper(), s)
    return s

def main(args):
    # Modèle lettres
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modèle landmarks introuvable. Entraîne d'abord src/train_landmarks_mlp.py.")
    classes = json.load(open(LABELS_JSON,"r",encoding="utf-8"))
    clf = joblib.load(MODEL_PATH)

    # Vocabulaire pour autocorrection
    vocab = build_vocab(args.lang, size=args.vocab_size)

    # Webcam + MediaPipe
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): raise RuntimeError("Webcam introuvable")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           model_complexity=1,
                           min_detection_confidence=args.det_conf,
                           min_tracking_confidence=args.trk_conf)

    # (Optionnel) segmentation personne
    seg = None
    if args.use_seg:
        try:
            seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            seg = None  # continue sans segmentation

    # Dessin skelette (option)
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    mirror=True
    ring = deque(maxlen=args.stable_frames)
    last_letter_time = 0.0
    word_chars: list[str] = []
    phrase_tokens: list[str] = []
    nohand_ms = 0.0
    prev_t = time.monotonic()

    help_line = "Q quit | M mirror | BKSP del | ENTER commit | SPACE space | . , ! ? ; : punct"

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if mirror: frame = cv2.flip(frame,1)
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t = time.monotonic(); dt = t - prev_t; prev_t = t

            res = hands.process(rgb)
            pred_txt = "-"; p_txt = 0.0

            if res.multi_hand_landmarks:
                nohand_ms = 0.0
                hand_lms = res.multi_hand_landmarks[0]
                lms_xy = np.array([[lm.x, lm.y] for lm in hand_lms.landmark], dtype=np.float32)
                handed = None
                try:
                    handed = res.multi_handedness[0].classification[0].label  # 'Left'/'Right'
                except Exception:
                    pass

                # ---- Masque convexe + (option) segmentation personne ----
                xs = (lms_xy[:,0]*W).astype(int); ys = (lms_xy[:,1]*H).astype(int)
                pts = list(zip(xs, ys))
                mask_hull, hull = convex_hull_mask(
                    (W,H), pts, dilate_px=args.mask_dilate, feather=args.mask_feather
                )
                if seg is not None:
                    seg_mask = seg.process(rgb).segmentation_mask
                    seg_bin = (seg_mask > args.seg_thresh).astype(np.uint8) * 255
                    seg_bin = cv2.GaussianBlur(seg_bin, (11,11), 0)
                    mask = cv2.bitwise_and(mask_hull, seg_bin)
                else:
                    mask = mask_hull

                frame = focus_effect(frame, mask, darken=args.darken, inside_boost=args.inside_boost)

                # ---- Prédiction lettre (landmarks -> MLP) ----
                feats = build_feature_vector(lms_xy, handed).reshape(1,-1)
                prob = clf.predict_proba(feats)[0]
                idx = int(prob.argmax()); p = float(prob[idx])
                pred = classes[idx]
                pred_txt, p_txt = pred, p
                if p >= args.min_prob:
                    ring.append(pred)

                # ---- Dessins shape ----
                if hull is not None:
                    cv2.polylines(frame, [hull], True, (0,255,255), 2, cv2.LINE_AA)
                if args.draw_skeleton:
                    mp_draw.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
            else:
                nohand_ms += dt*1000

            # ---- Validation lettre (fenêtre + cooldown) ----
            now_ms = time.monotonic()*1000
            if len(ring)==ring.maxlen and (now_ms - last_letter_time) >= args.letter_cooldown_ms:
                maj = Counter(ring).most_common(1)[0][0]
                if not word_chars or word_chars[-1] != maj:
                    word_chars.append(maj); last_letter_time = now_ms
                ring.clear()

            # ---- Fin de mot par pause sans main ----
            if nohand_ms >= args.word_pause_ms and word_chars:
                raw = "".join(word_chars)
                corrected = autocorrect(raw, vocab, threshold=args.autocorr_thresh)
                phrase_tokens.append(corrected)
                word_chars.clear()
                nohand_ms = 0.0

            # ---- HUD ----
            fps = 1.0/max(1e-6, dt)
            put_text(frame, f"FPS {fps:.1f}", (10,20), 0.6, 1)
            put_text(frame, f"Letter {pred_txt} ({p_txt:.2f})", (10,45))
            put_text(frame, f"Word: {''.join(word_chars)}", (10,75))

            phrase_preview = tidy_phrase(phrase_tokens + (["".join(word_chars)] if word_chars else []))
            put_text(frame, f"Phrase: {phrase_preview[:70]}", (10,105), 0.6, 2)
            put_text(frame, help_line, (10,H-10), 0.5, 1)

            cv2.imshow("Letters → Words → Sentence (landmarks + MLP + XR focus)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'),27): break
            elif k in (ord('m'),ord('M')): mirror = not mirror
            elif k==8 and word_chars:      # Backspace (lettre)
                word_chars.pop()
            elif k==13:                    # ENTER → commit mot
                if word_chars:
                    raw = "".join(word_chars)
                    corrected = autocorrect(raw, vocab, threshold=args.autocorr_thresh)
                    phrase_tokens.append(corrected)
                    word_chars.clear()
            elif k==32:                    # SPACE
                if word_chars:
                    raw = "".join(word_chars)
                    corrected = autocorrect(raw, vocab, threshold=args.autocorr_thresh)
                    phrase_tokens.append(corrected)
                    word_chars.clear()
                phrase_tokens.append("")    # espace explicite
            elif k in map(ord, [".",",","!","?",";",":"]):
                ch = chr(k)
                if word_chars:
                    raw = "".join(word_chars)
                    corrected = autocorrect(raw, vocab, threshold=args.autocorr_thresh)
                    phrase_tokens.append(corrected)
                    word_chars.clear()
                phrase_tokens.append(ch)

    finally:
        cap.release(); cv2.destroyAllWindows(); hands.close()
        try:
            if seg: del seg
        except Exception:
            pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Détection & tracking
    ap.add_argument("--det_conf", type=float, default=0.6)
    ap.add_argument("--trk_conf", type=float, default=0.6)
    # Stabilisation/tempo
    ap.add_argument("--min_prob", type=float, default=0.80)
    ap.add_argument("--stable_frames", type=int, default=10)
    ap.add_argument("--letter_cooldown_ms", type=int, default=1000)
    ap.add_argument("--word_pause_ms", type=int, default=1500)
    # Langue / correction
    ap.add_argument("--lang", type=str, default="fr", choices=["fr","en"])
    ap.add_argument("--vocab_size", type=int, default=50000)
    ap.add_argument("--autocorr_thresh", type=int, default=85)
    # Effet “radiographie”
    ap.add_argument("--darken", type=float, default=0.20, help="Assombrissement du fond (0..1)")
    ap.add_argument("--inside_boost", type=float, default=1.05, help="Boost léger sur la main")
    ap.add_argument("--mask_dilate", type=int, default=30, help="Dilatation du masque (px)")
    ap.add_argument("--mask_feather", type=int, default=21, help="Adoucissement du bord (px)")
    ap.add_argument("--use_seg", action="store_true", default=False, help="Active Selfie Segmentation")
    ap.add_argument("--seg_thresh", type=float, default=0.5, help="Seuil binaire segmentation (0..1)")
    ap.add_argument("--draw_skeleton", action="store_true", default=True, help="Dessine squelette MediaPipe")
    args = ap.parse_args()
    main(args)
