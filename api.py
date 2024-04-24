from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('model_file_30epochs.h5')

# Load the face detector
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Labels dictionary
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the image file in grayscale
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

        # Detect face in the image
        faces = faceDetect.detectMultiScale(img, 1.3, 5)

        # If no faces are detected, return an error message
        if len(faces) == 0:
            return jsonify({"error": "No faces found"}), 400

        # For simplicity, we'll only consider the first face found
        for (x, y, w, h) in faces:
            sub_face_img = img[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            # Build the response
            response = {
                "emotion": labels_dict[label]
            }
            return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 8080
    app.run(port=3000, debug=True)
