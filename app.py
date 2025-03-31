from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import cv2
import numpy as np
import face_recognition
import requests

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("service.json")  # Ensure this file is in your project folder
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/process-images', methods=['POST'])
def process_images():
    try:
        # Fetch all users from the "dataset" collection
        users = db.collection("dataset").stream()

        for user_doc in users:
            user_id = user_doc.id  # User ID
            images_ref = db.collection("dataset").document(user_id).collection("images")

            # Fetch images from user's "images" subcollection
            images = images_ref.stream()

            for image_doc in images:
                image_data = image_doc.to_dict()
                base64_string = image_data.get("image_base64")

                if not base64_string:
                    continue  # Skip if no image data

                # Decode Base64 to Image
                img_data = base64.b64decode(base64_string)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Resize image to optimize memory usage
                scale_percent = 50  # Reduce size to 50%
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                img = cv2.resize(img, (width, height))

                # Convert Image to Face Encoding
                face_encodings = face_recognition.face_encodings(img)
                if face_encodings:
                    encoding = face_encodings[0].tolist()
                else:
                    continue  # Skip if no face found

                # Send Encoding to Raspberry Pi
                raspberry_pi_url = "http://100.67.213.11:5000/receive-encoding"  # Replace with actual Pi IP
                response = requests.post(raspberry_pi_url, json={"user_id": user_id, "encoding": encoding})

                # Delete processed image from Firestore
                images_ref.document(image_doc.id).delete()

        return jsonify({"message": "All images processed"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
