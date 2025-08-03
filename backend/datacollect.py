import cv2
import csv
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

labels_map = {
    'h': 'Hello',
    'o': 'Okay',
    'g': 'Good job',
    't': 'Thank you',
    'y': 'Yes'
}

output_file = 'landmarks_dataset.csv'

cap = cv2.VideoCapture(0)

print("[INFO] Press:")
for k, v in labels_map.items():
    print(f"  '{k}' to record '{v}'")
print("  'q' to quit")

with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])

    while True:
        success, img = cap.read()
        if not success:
            print("Camera error.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Landmark Collector", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif chr(key) in labels_map:
            label = labels_map[chr(key)]
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in hand.landmark]
                flat = [coord for point in landmarks for coord in point]
                row = [label] + flat
                writer.writerow(row)
                print(f"[+] Saved: {label}")

cap.release()
cv2.destroyAllWindows()
