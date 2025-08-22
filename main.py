# main.py — Webcam -> main isolée -> 28x28 -> CNN lettres -> mots -> phrase
# Usage: uv run python main.py --weights outputs/models/asl_cnn.pt
from __future__ import annotations
import argparse, time, math, json
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

# --------- Chargement des classes (ordre EXACT du training) ----------
def load_class_names(default=None):
    # Essaie de lire un mapping exporté pendant l'entraînement (recommandé)
    candidates = [
        "outputs/models/asl_class_names.json",
        "outputs/models/gesture_labels.json"
    ]
    for p in candidates:
        pth = Path(p)
        if pth.exists():
            try:
                return json.load(open(pth, "r", encoding="utf-8"))
            except Exception:
                pass
    return default or ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

CLASS_NAMES = load_class_names()

# --------- Modèle identique à l'entraînement ----------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128,num_classes))
    def forward(self, x): return self.head(self.features(x))

# --------- Utils visuels ----------
def put_text(img, text, org, scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

def convex_hull_mask(shape_wh, pts_xy, dilate_px=30, feather=21):
    h, w = shape_wh[1], shape_wh[0]
    mask = np.zeros((h, w), np.uint8)
    if not pts_xy:
        return mask
    hull = cv2.convexHull(np.array(pts_xy, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, iterations=1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather|1, feather|1), 0)
    return mask, hull

def focus_effect(frame, mask, darken=0.15, inside_boost=1.05):
    mask_f = (mask.astype(np.float32)/255.0)[:, :, None]
    outside = (frame.astype(np.float32) * darken).astype(np.uint8)
    inside  = np.clip(frame.astype(np.float32) * inside_boost, 0, 255).astype(np.uint8)
    out = (inside * mask_f + outside * (1.0 - mask_f)).astype(np.uint8)
    return out

def expand_square(x1,y1,x2,y2,w,h,f=1.40):
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    side = int(max(x2-x1, y2-y1) * f)
    nx1 = int(max(0, cx - side/2)); ny1 = int(max(0, cy - side/2))
    nx2 = int(min(w-1, cx + side/2)); ny2 = int(min(h-1, cy + side/2))
    return nx1, ny1, nx2, ny2

# --------- Normalisation orientation + main gauche/droite ----------
def compute_hand_angle_deg(lms, w, h):
    WRIST = 0; MIDDLE_MCP = 9
    wx, wy = lms[WRIST].x*w, lms[WRIST].y*h
    mx, my = lms[MIDDLE_MCP].x*w, lms[MIDDLE_MCP].y*h
    ang = math.degrees(math.atan2(my - wy, mx - wx))  # [-180..180]
    # On veut "vertical" = 90° (vers le haut). Rotation à appliquer = 90 - ang
    return 90.0 - ang

def rotate_image(image, angle_deg):
    h, w = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

# --------- Prétraitement MNIST-like ----------
def to_mnist_like(roi_bgr, clahe_clip=2.0, clahe_grid=8, show_dbg=False):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (5,5), 0)
    _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if thr.mean() > 127:  # on veut "main claire sur fond sombre"
        thr = 255 - thr
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    patch = thr[y:y+h, x:x+w]
    side = max(w,h)
    canvas = np.zeros((side,side), np.uint8)
    ox, oy = (side-w)//2, (side-h)//2
    canvas[oy:oy+h, ox:ox+w] = patch
    img28 = cv2.resize(canvas, (28,28), interpolation=cv2.INTER_AREA)
    img28 = (img28.astype(np.float32)/255.0)[None, :, :]  # (1,28,28)
    dbg = None
    if show_dbg:
        dbg = cv2.resize((img28[0]*255).astype(np.uint8),(112,112),interpolation=cv2.INTER_NEAREST)
        dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
    return img28, dbg

# --------- Qualité & contrôle taille/main centrée ----------
def laplacian_sharpness(gray_roi):
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var()

def quality_gate(frame_w, frame_h, roi_box, min_side_px=120, center_tol=0.25):
    x1,y1,x2,y2 = roi_box
    side = max(x2-x1, y2-y1)
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    # Taille minimale du carré (évite "main trop loin")
    size_ok = side >= min_side_px
    # Centralité (à ~25% près de l'écran)
    dx = abs(cx - frame_w/2) / (frame_w/2)
    dy = abs(cy - frame_h/2) / (frame_h/2)
    center_ok = (dx <= center_tol) and (dy <= center_tol)
    return size_ok, center_ok, int(side)

def draw_quality_hud(frame, size_ok, center_ok, side_px, min_side_px):
    h, w = frame.shape[:2]
    msg = []
    if not size_ok:  msg.append(f"Main trop loin : {side_px}px < {min_side_px}px — approchez-vous")
    if not center_ok: msg.append("Centrez la main")
    if msg:
        y = 140
        for m in msg:
            put_text(frame, m, (10, y), scale=0.7, thickness=2); y += 28

# --------- Multi-échelle optionnelle (petit boost robustesse) ----------
def best_mnist_patch(model, device, roi_bgr, scales=(1.0, 1.15, 0.9), show_dbg=False):
    best = None
    best_prob = -1.0
    best_dbg = None
    for s in scales:
        if s != 1.0:
            h, w = roi_bgr.shape[:2]
            ns = (max(16, int(w*s)), max(16, int(h*s)))
            scaled = cv2.resize(roi_bgr, ns, interpolation=cv2.INTER_LINEAR)
            # recadrer au centre à la taille d'origine
            x0 = max(0, (scaled.shape[1]-w)//2)
            y0 = max(0, (scaled.shape[0]-h)//2)
            scaled = scaled[y0:y0+h, x0:x0+w]
        else:
            scaled = roi_bgr
        img28, dbg = to_mnist_like(scaled, show_dbg=show_dbg)
        if img28 is None: 
            continue
        with torch.no_grad():
            logits = model(torch.from_numpy(img28).unsqueeze(0).to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        p = float(prob.max()); idx = int(prob.argmax())
        if p > best_prob:
            best_prob, best = p, (idx, prob)
            if show_dbg: best_dbg = dbg
    return best, best_dbg  # ((idx, prob_array), dbg)

# --------- Main loop ----------
def main(args):
    # Chargement modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(len(CLASS_NAMES)).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state); model.eval()

    # Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): raise RuntimeError("Webcam introuvable")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           model_complexity=1, min_detection_confidence=args.det_conf, min_tracking_confidence=args.trk_conf)

    # (Optionnel) Selfie Segmentation — si dispo, on la branche
    selfie = None
    try:
        mp_seg = mp.solutions.selfie_segmentation
        selfie = mp_seg.SelfieSegmentation(model_selection=1)
    except Exception:
        selfie = None

    # Hyper-paramètres (tempo/stabilité & focus)
    MIN_PROB = args.min_prob
    MIN_STABLE_FRAMES = args.stable_frames
    LETTER_COOLDOWN_MS = args.letter_cooldown_ms
    WORD_PAUSE_MS = args.word_pause_ms
    MIN_ROI_SIDE_PX = args.min_roi_side_px
    DARKEN = args.darken
    INSIDE_BOOST = args.inside_boost
    SHOW_DBG = args.show_dbg

    mirror=True
    ring=deque(maxlen=MIN_STABLE_FRAMES)
    last_letter_time = 0.0
    word, phrase = [], []
    nohand_ms=0.0; prev_t=time.monotonic()
    screenshot_id=0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if mirror: frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            t = time.monotonic(); dt = t - prev_t; prev_t = t

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            letter_disp='-'; p_disp=0.0; dbg_patch=None
            hand_present=False
            if res.multi_hand_landmarks:
                hand_present=True; nohand_ms=0.0
                lms = res.multi_hand_landmarks[0].landmark
                pts = [(int(l.x*w), int(l.y*h)) for l in lms]

                # Masque convexe + (option) segmentation personne
                mask_hull, hull = convex_hull_mask((w,h), pts, dilate_px=30, feather=21)
                if selfie is not None:
                    seg = selfie.process(rgb).segmentation_mask
                    seg_bin = (seg > 0.5).astype(np.uint8) * 255
                    seg_bin = cv2.GaussianBlur(seg_bin, (11,11), 0)
                    mask_comb = cv2.bitwise_and(mask_hull, seg_bin)
                else:
                    mask_comb = mask_hull

                # Assombrir le fond (effet "radiographie")
                frame = focus_effect(frame, mask_comb, darken=DARKEN, inside_boost=INSIDE_BOOST)

                # Boîte ROI autour de la main
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                x1,y1 = max(min(xs),0), max(min(ys),0)
                x2,y2 = min(max(xs),w-1), min(max(ys),h-1)
                x1,y1,x2,y2 = expand_square(x1,y1,x2,y2,w,h,f=1.40)

                # Contrôle qualité (taille/centrage)
                size_ok, center_ok, side_px = quality_gate(w, h, (x1,y1,x2,y2), min_side_px=MIN_ROI_SIDE_PX, center_tol=0.25)
                draw_quality_hud(frame, size_ok, center_ok, side_px, MIN_ROI_SIDE_PX)

                # Si pas OK -> pas de prédiction, juste HUD d'aide
                if not (size_ok and center_ok):
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2,cv2.LINE_AA)
                    # affiche quand même la boîte et saute la prédiction
                    cv2.imshow("ASL Letters -> Word -> Phrase (focused)", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord('q'),27): break
                    elif k in (ord('m'),ord('M')): mirror = not mirror
                    elif k in (ord('s'),ord('S')):
                        Path("outputs/figures").mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(f"outputs/figures/frame_{screenshot_id:03d}.png", frame); screenshot_id+=1
                    elif k==8 and word: word.pop()
                    elif k==13 and word: phrase.append("".join(word)); word.clear()
                    elif k==32: phrase.append("")  # espace manuel
                    continue

                roi = frame[y1:y2, x1:x2]

                # Normaliser main gauche/droite : si "Left", miroir pour uniformiser
                try:
                    label = res.multi_handedness[0].classification[0].label
                    if label.lower().startswith('left'):
                        roi = cv2.flip(roi, 1)
                except Exception:
                    pass

                # Normaliser orientation (poignet -> MCP majeur vertical)
                angle = compute_hand_angle_deg(lms, w, h)
                roi = rotate_image(roi, angle)

                # Prédiction (best de 3 échelles)
                best, dbg_patch = best_mnist_patch(model, device, roi, scales=(1.0, 1.15, 0.9), show_dbg=SHOW_DBG)
                if best is not None:
                    idx, prob = best
                    p = float(prob[idx]); letter = CLASS_NAMES[idx]
                    letter_disp, p_disp = letter, p
                    if p >= MIN_PROB:
                        ring.append(letter)

                # Dessins
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,130,0),2,cv2.LINE_AA)
                cv2.polylines(frame, [cv2.convexHull(np.array(pts, np.int32))], True, (0,255,255), 2, cv2.LINE_AA)

            else:
                nohand_ms += dt*1000

            # Stabilisation + cooldown (validation d'une lettre)
            now_ms = time.monotonic()*1000
            if len(ring)==ring.maxlen and (now_ms - last_letter_time) >= LETTER_COOLDOWN_MS:
                maj = Counter(ring).most_common(1)[0][0]
                if not word or word[-1] != maj:
                    word.append(maj); last_letter_time = now_ms
                ring.clear()

            # Fin de mot par pause sans main
            if nohand_ms >= WORD_PAUSE_MS and word:
                phrase.append("".join(word)); word.clear(); nohand_ms=0.0

            # HUD
            fps = 1.0/max(1e-6, dt)
            put_text(frame,f"FPS {fps:.1f}",(10,20),0.6,1)
            put_text(frame,f"Pred {letter_disp} ({p_disp:.2f})",(10,45))
            put_text(frame,f"Word: {''.join(word)}",(10,75))
            put_text(frame,f"Phrase: {' '.join([w for w in phrase if w])[:70]}",(10,105))
            put_text(frame,"Q quit | M mirror | S shot | BKSP del | ENTER commit word | SPACE space",(10,h-10),0.5,1)

            # Aperçu 28x28
            if SHOW_DBG and dbg_patch is not None:
                y0=10; x0=frame.shape[1]-122
                cv2.rectangle(frame,(x0-6,y0-6),(frame.shape[1]-10,y0+122),(0,0,0),2)
                frame[y0:y0+dbg_patch.shape[0], x0:x0+dbg_patch.shape[1]] = dbg_patch

            cv2.imshow("ASL Letters -> Word -> Phrase (focused)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'),27): break
            elif k in (ord('m'),ord('M')): mirror = not mirror
            elif k in (ord('s'),ord('S')):
                Path("outputs/figures").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"outputs/figures/frame_{screenshot_id:03d}.png", frame); screenshot_id+=1
            elif k==8 and word: word.pop()
            elif k==13 and word: phrase.append("".join(word)); word.clear()
            elif k==32: phrase.append("")  # espace manuel

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        try:
            if selfie: del selfie
        except Exception:
            pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="outputs/models/asl_cnn.pt")
    # Robustesse détection
    ap.add_argument("--det_conf", type=float, default=0.6)
    ap.add_argument("--trk_conf", type=float, default=0.6)
    # Tempo/stabilité
    ap.add_argument("--min_prob", type=float, default=0.85)
    ap.add_argument("--stable_frames", type=int, default=10)
    ap.add_argument("--letter_cooldown_ms", type=int, default=1200)
    ap.add_argument("--word_pause_ms", type=int, default=1600)
    # Focus & taille minimale
    ap.add_argument("--min_roi_side_px", type=int, default=120, help="Taille min du carré main en pixels; sinon popup 'approchez-vous'")
    ap.add_argument("--darken", type=float, default=0.15)
    ap.add_argument("--inside_boost", type=float, default=1.05)
    ap.add_argument("--show_dbg", action="store_true", default=True)
    args = ap.parse_args()
    main(args)
