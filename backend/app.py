from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load your trained model and labels
model = load_model(r"D:\Projects\Hand Sign Launguage\protottype\backend\models\landmark_model.keras")
with open(r"D:\Projects\Hand Sign Launguage\protottype\backend\models\labels.txt", "r") as f:
    labels = [line.strip().split(maxsplit=1)[1] for line in f]

# Initialize hand detector
detector = HandDetector(maxHands=1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read image from the uploaded file
        image_file = request.files['image']
        npimg = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Delay to avoid MediaPipe timestamp mismatch error
        time.sleep(0.01)

        # Detect hand and get landmarks
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            if lmList:
                landmark_data = []
                for lm in lmList:
                    landmark_data.extend([lm[0], lm[1]])

                prediction = model.predict(np.array([landmark_data]))
                label_index = np.argmax(prediction)
                confidence = float(np.max(prediction))
                label = labels[label_index]

                return jsonify({'prediction': label, 'confidence': confidence})

        return jsonify({'prediction': 'No hand detected', 'confidence': 0.0})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
