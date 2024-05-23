from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load your model (update the path as needed)
model = load_model('./dataset/model/New Plant Diseases Model dengan Sequential-Plant Disease Sequential-95.58.h5')

plant_names = [
    "Apple___Cedar_apple_rust",
    "Apple Healty",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Update target_size sesuai kebutuhan model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi jika diperlukan

    prediction = model.predict(img_array)
    # Lakukan decoding hasil prediksi jika diperlukan
    # Misalnya, jika model Anda mengklasifikasikan 3 kelas
    predicted_class = plant_names[np.argmax(prediction)]
    return predicted_class
