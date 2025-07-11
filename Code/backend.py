import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from PIL import Image
import pytesseract

model_dir = "neural-chat/INT8"
device = "CPU"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
ov_model = OVModelForCausalLM.from_pretrained(model_dir, device=device, compile=False)

# Compile the model once manually (this will stay in memory)
print("Compiling the model for OpenVINO...")
ov_model.compile()
print("Model compiled and ready.")

# === Flask setup ===
app = Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)

# Create upload directory if not exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# === Routes ===

# Serve frontend
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "frontend.html")

# Serve any static file (CSS, JS, images, etc.)
@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Handle text question
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "No question provided."}), 400

    try:
        inputs = tokenizer(question, return_tensors="pt")
        outputs = ov_model.generate(**inputs, max_new_tokens=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"answer": response.strip()})
    except Exception as e:
        print("Error:", e)
        return jsonify({"answer": "Error processing question."}), 500

# Handle image upload and OCR using Tesseract and forward to chat model
@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"answer": "No image uploaded."}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"answer": "No image selected."}), 400

    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(image_path)

    # OCR using Tesseract
    try:
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img, lang="eng").strip()

        if not extracted_text:
            return jsonify({"answer": "No text found in image."})

        # Send extracted text to the language model for response
        inputs = tokenizer(extracted_text, return_tensors="pt")
        outputs = ov_model.generate(**inputs, max_new_tokens=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({
            "answer": response.strip(),
            "extracted_text": extracted_text
        })

    except Exception as e:
        print("Tesseract OCR or Chat Error:", e)
        return jsonify({"answer": "Error reading or processing text from image."}), 500

# === Start the server ===
if __name__ == "__main__":
    app.run(debug=True, port=3000, threaded=True)