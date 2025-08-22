 # src/demo_realtime.py
# Démo temps réel : détection de la main, landmarks, handedness, FPS.
# Dépendances : mediapipe==0.10.21, opencv-python
# Lancer : uv run python src/demo_realtime.py

import cv2
import time
import numpy as np
import mediapipe as mp

def put_text(img, text, org, scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_landmarks(image, hand_landmarks):
    h, w = image.shape[:2]
    # Dessiner les points (21) et segments principaux
    idx_pairs = [
        (0,1),(1,2),(2,3),(3,4),      # pouce
        (0,5),(5,6),(6,7),(7,8),      # index
        (0,9),(9,10),(10,11),(11,12), # majeur
        (0,13),(13,14),(14,15),(15,16), # annulaire
        (0,17),(17,18),(18,19),(19,20)  # auriculaire
    ]
    pts = []
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x,y))
        cv2.circle(image, (x,y), 3, (0,255,0), -1, cv2.LINE_AA)
    for i,j in idx_pairs:
        cv2.line(image, pts[i], pts[j], (0,200,255), 2, cv2.LINE_AA)

def main():
    # Webcam (0 = défaut). Si laptop + webcam USB, essaye 1 ou 2.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam (index 0).")

    # Taille d’image raisonnable pour de bonnes perfs CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,   # 0 (plus léger) à 1 (plus précis)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    prev_t = time.time()
    show_bbox = True
    mirror = True
    screenshot_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Option miroir pour une sensation "selfie"
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now

            if result.multi_hand_landmarks:
                for i, hand_lms in enumerate(result.multi_hand_landmarks):
                    # Dessin landmarks
                    draw_landmarks(frame, hand_lms)

                # Handedness (droite/gauche + score)
                if result.multi_handedness:
                    for i, handed in enumerate(result.multi_handedness):
                        label = handed.classification[0].label  # 'Right' ou 'Left'
                        score = handed.classification[0].score
                        put_text(frame, f"{label} ({score:.2f})", (10, 40 + 25*i))

                # BBox (approximative) par main
                if show_bbox:
                    h, w = frame.shape[:2]
                    for hand_lms in result.multi_hand_landmarks:
                        xs = [int(lm.x * w) for lm in hand_lms.landmark]
                        ys = [int(lm.y * h) for lm in hand_lms.landmark]
                        x1, y1 = max(min(xs), 0), max(min(ys), 0)
                        x2, y2 = min(max(xs), w-1), min(max(ys), h-1)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 130, 0), 2, cv2.LINE_AA)

            # Overlays
            put_text(frame, f"FPS: {fps:.1f}", (10, 20))
            put_text(frame, "Keys: [Q]uit  [B]box on/off  [M]irror on/off  [S]creenshot", (10, frame.shape[0]-10), scale=0.5, thickness=1)

            cv2.imshow("Mediapipe Hands — Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q ou ESC
                break
            elif key in (ord('b'), ord('B')):
                show_bbox = not show_bbox
            elif key in (ord('m'), ord('M')):
                mirror = not mirror
            elif key in (ord('s'), ord('S')):
                fn = f"outputs/figures/screenshot_{screenshot_id:03d}.png"
                cv2.imwrite(fn, frame)
                screenshot_id += 1
                print(f"[INFO] Screenshot sauvegardé: {fn}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
 