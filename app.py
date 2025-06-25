from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import torch, io, os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from werkzeug.utils import secure_filename
import webbrowser
import threading
import random
import cv2
import numpy as np


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model_id = "dima806/attractive_faces_celebs_detection"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)


# Check for face using OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        image_url = "/static/blank.png"
        return render_template("no_face.html", image_url=image_url)


    # Run attractiveness model
    try:
        image = Image.open(filepath).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        score_raw = probs[model.config.label2id["attractive"]].item() * 100

        # Adjust score if too low
        if score_raw < 65:
            score = random.randint(65, 85)
        else:
            score = int(round(score_raw))

        # Compliment based on score
        if score >= 86:
            compliment = "You're absolutely stunning! 🌟"
        elif score >= 70:
            compliment = "Looking great — definitely photogenic! 😊"
        else:
            compliment = "Nice! You've got a natural charm. ✨"

        image_url = f"/static/uploads/{filename}"
        return render_template("result.html", score=score, image_url=image_url, compliment=compliment)

    except Exception as e:
        print("Error processing image:", e)
        return "Processing error", 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True)
