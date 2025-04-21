from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Chargement des deux modèles
model_vgg16 = load_model('vgg16_breast_model.h5')
model_vgg19 = load_model('model_vgg19.keras')

# Noms des classes
class_names = ['benign', 'malignant', 'normal']

# Fonction de prétraitement de l'image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Route principale
@app.route('/')
def index():
    return render_template('index.html')

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Aucune image envoyée"

    file = request.files['image']
    if file.filename == '':
        return "Fichier vide"

    # Sauvegarde de l'image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Prétraitement
    img = preprocess_image(filepath)

    # Choix du modèle
    model_choice = request.form.get('model')
    model = model_vgg16 if model_choice == 'vgg16' else model_vgg19

    # Prédiction
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100


    return render_template(
        'index.html',
        prediction=predicted_class,
        confidence=confidence,
        image_path=filepath
    )

# Lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)
