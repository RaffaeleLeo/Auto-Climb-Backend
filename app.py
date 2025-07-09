from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import requests
from PIL import ImageOps
import datetime

app = Flask(__name__)
model = YOLO("weights/simple_model_2.onnx")  # or .pt if preferred

headers = {
    "User-Agent": "Mozilla/5.0"  # This tricks Imgur into thinking you're a browser
}

@app.route('/detect', methods=['POST'])
def detect():
    
    stats = {
        'timestamp': datetime.datetime.now().isoformat(),
        'image_source': None,
        'image_dimensions': None,
        'detection_count': 0,
        'speed': None,
        'average_confidence': 0
    }

    # Option 1: If image is uploaded directly as a file
    if 'image' in request.files:
        file = request.files['image']
        img = Image.open(file.stream)

    # Option 2: If image is passed via a URL in the JSON payload
    elif request.is_json and 'image_url' in request.json:
        image_url = request.json['image_url']
        try:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            stats['image_source'] = 'url'
        except Exception as e:
            return jsonify({"error": f"Failed to fetch image from URL: {str(e)}"}), 400

    else:
        return jsonify({"error": "No image file or URL provided"}), 400

    img = ImageOps.exif_transpose(img).convert("RGB")
    orig_w, orig_h = img.size
    stats['image_dimensions'] = {'width': orig_w, 'height': orig_h}

    results = model(img)
    boxes = results[0].boxes  # ultralytics.engine.results.Boxes
    stats['speed'] = results[0].speed
    stats['total_processing_time'] = sum(results[0].speed.values())

    detections = []
    confidences = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = round(float(box.conf[0]), 3)
        bbox = [x1, y1, x2 - x1, y2 - y1]  # convert to x, y, width, height
        detections.append({
            "bbox": bbox,
            "confidence": round(float(box.conf[0]), 3),
            "class": int(box.cls[0]),
            "imageWidth": orig_w,
            "imageHeight": orig_h,
        })
        confidences.append(confidence)
    stats['detection_count'] = len(detections)
    stats['average_confidence'] = round(sum(confidences)/len(confidences), 3) if confidences else 0

    return jsonify({
        "detections": detections,
        "stats": stats
    })

if __name__ == "__main__":
    app.run(debug=True)