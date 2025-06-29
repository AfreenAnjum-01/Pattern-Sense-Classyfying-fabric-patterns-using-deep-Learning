from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("fabric_cnn_model.h5")

# Define class names based on your model training (example: 3 classes)
class_names = ['floral', 'striped', 'polka_dot']  # update as per your labels

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400

    if file:
        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        # Preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        pred = model.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]

        return render_template('result.html', prediction=predicted_class, image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
