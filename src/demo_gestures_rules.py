import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque, Counter

# Index des landmarks Mediapipe Hands
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}
THRESH = {
    "extended_margin": 0.035,   # marge (normale) pour décider "étendu vs replié"
    "pinch_dist": 0.06,         # distance (normale) pouce-index pour "pinch"
    "thumb_up_y": 0.04,         # pouce pointe "vers le haut" si TIP au-dessus du MCP de > ce delta
}

def put_text(img, text, org, scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

def norm_dist(a, b):
    # distance Euclidienne en coordonnées normalisées (x,y)
    dx, dy = a.x - b.x, a.y - b.y
    return np.hypot(dx, dy)

def is_finger_extended(lm, wrist, mcp, pip, dip, tip):
    """Heuristique robuste : doigt 'étendu' si TIP est sensiblement plus loin du poignet que PIP/DIP."""
    d_tip = norm_dist(lm[tip], wrist)
    d_pip = norm_dist(lm[pip], wrist)
    d_dip = norm_dist(lm[dip], wrist)
    # étendu si le TIP dépasse le PIP/DIP d'une certaine marge
    return (d_tip > d_pip + THRESH["extended_margin"]) and (d_tip > d_dip + THRESH["extended_margin"])

def is_thumb_extended_up(lm):
    """Pouce étendu ET pointe vers le haut approximativement (y plus petit = vers le haut sur l'image)."""
    # étendu si TIP plus loin du poignet que IP/MCP
    wrist = lm[WRIST]
    d_tip = norm_dist(lm[THUMB_TIP], wrist)
    d_ip  = norm_dist(lm[THUMB_IP], wrist)
    d_mcp = norm_dist(lm[THUMB_MCP], wrist)
    extended = (d_tip > d_ip + THRESH["extended_margin"]) and (d_tip > d_mcp + THRESH["extended_margin"])

    # "vers le haut" si le TIP est significativement au-dessus du MCP
    up = (lm[THUMB_TIP].y + THRESH["thumb_up_y"] < lm[THUMB_MCP].y)
    return extended, up

def classify_gesture(lm):
    """Retourne (label, score_conf) basé sur des règles simples."""
    wrist = lm[WRIST]

    # Détections doigts (hors pouce)
    fingers_state = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        fingers_state[name] = is_finger_extended(lm, wrist, mcp, pip, dip, tip)

    # Pinch: distance pouce-index TIP très faible
    pinch = norm_dist(lm[THUMB_TIP], lm[INDEX_TIP]) < THRESH["pinch_dist"]

    thumb_ext, thumb_up = is_thumb_extended_up(lm)

    # Règles (ordre important)
    # 1) Pinch si distance très faible et autres doigts plutôt repliés
    if pinch and (not fingers_state["middle"]) and (not fingers_state["ring"]) and (not fingers_state["pinky"]):
        return "Pinch", 0.95

    # 2) Point: index étendu, les autres (middle, ring, pinky) repliés
    if fingers_state["index"] and not (fingers_state["middle"] or fingers_state["ring"] or fingers_state["pinky"]):
        return "Point", 0.90

    # 3) ThumbsUp: pouce étendu ET "up", autres doigts repliés
    if thumb_ext and thumb_up and not (fingers_state["index"] or fingers_state["middle"] or fingers_state["ring"] or fingers_state["pinky"]):
        return "ThumbsUp", 0.92

    # 4) Fist: tout replié (pouce peut être à l'intérieur)
    if (not fingers_state["index"] and not fingers_state["middle"] and
        not fingers_state["ring"]  and not fingers_state["pinky"] and not thumb_ext):
        return "Fist", 0.85

    # 5) Open: doigts (index, middle, ring, pinky) étendus (pouce indifferent)
    if fingers_state["index"] and fingers_state["middle"] and fingers_state["ring"] and fingers_state["pinky"]:
        return "Open", 0.88

    # Sinon inconnu
    return "Unknown", 0.50

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
        cv2.circle(image, (x,y), 3, (0,255,0), -1, cv2.LINE_AA)
    for i,j in idx_pairs:
        cv2.line(image, pts[i], pts[j], (0,200,255), 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam (index 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Lissage du label pour un rendu stable
    label_buffer = deque(maxlen=8)

    prev_t = time.time()
    mirror = True
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now

            label_display = "No hand"
            score = 0.0

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                draw_landmarks(frame, hand_lms)

                # classification par règles
                label, score = classify_gesture(hand_lms.landmark)
                label_buffer.append(label)

                # vote majoritaire sur les N derniers
                if len(label_buffer) >= 3:
                    label_counts = Counter(label_buffer)
                    label_display = label_counts.most_common(1)[0][0]
                else:
                    label_display = label

            put_text(frame, f"Gesture: {label_display}", (10, 40))
            put_text(frame, f"FPS: {fps:.1f}", (10, 20))
            put_text(frame, "Keys: [Q]uit  [M]irror on/off", (10, frame.shape[0]-10), scale=0.6, thickness=1)

            cv2.imshow("Mediapipe Hands — Gestures (rules)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('m'), ord('M')):
                mirror = not mirror

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
