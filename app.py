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
cred = credentials.Certificate("/home/prankit/Downloads/service.json")  # Ensure this file is in the project folder
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Fetch latest image document from Firestore
        docs = db.collection("images").limit(1).stream()
        doc_list = list(docs)

        if not doc_list:
            return jsonify({"error": "No images found"}), 404

        doc = doc_list[0]
        doc_id = doc.id
        data = doc.to_dict()
        base64_string = data.get("image_base64")

        if not base64_string:
            return jsonify({"error": "Image data missing"}), 400

        # Decode Base64 to Image
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert Image to Face Encoding
        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:
            encoding = face_encodings[0].tolist()  # Convert NumPy array to list
        else:
            return jsonify({"error": "No face detected"}), 400

        # Send Encoding to Raspberry Pi
        raspberry_pi_url = "http://100.67.213.11:5000/receive-encoding"  # Replace with actual Pi IP
        response = requests.post(raspberry_pi_url, json={"encoding": encoding})

        # Delete processed image from Firestore
        db.collection("images").document(doc_id).delete()

        return jsonify({"message": "Processed and sent encoding", "raspberry_response": response.json()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Change port if needed
