from flask import Flask, request, render_template, redirect, url_for, send_file
from PIL import Image
import torch, io, uuid, threading, webbrowser, random, cv2, numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)

# In-memory image store
image_cache = {}

# Load model and processor
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
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # OpenCV face detection
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Store image in memory with UUID
    image_id = str(uuid.uuid4())
    image_cache[image_id] = image_bytes

    if len(faces) == 0:
        return render_template("no_face.html", image_id=image_id)

    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        score_raw = probs[model.config.label2id["attractive"]].item() * 100
        score = int(round(score_raw)) if score_raw >= 65 else random.randint(65, 85)

        compliment = (
            "You're absolutely stunning! 🌟" if score >= 86 else
            "Looking great — definitely photogenic! 😊" if score >= 70 else
            "Nice! You've got a natural charm. ✨"
        )

        return render_template("result.html", score=score, image_id=image_id, compliment=compliment)

    except Exception as e:
        print("Error processing image:", e)
        return "Processing error", 500

@app.route("/image/<image_id>")
def serve_image(image_id):
    image_bytes = image_cache.get(image_id)
    if not image_bytes:
        return "Image not found", 404
    return send_file(io.BytesIO(image_bytes), mimetype="image/jpeg")

# Optional: remove unused cleanup and upload folder logic
# Optional: keep auto-browser opening for local development

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True)
