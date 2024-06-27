import PIL
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load your model (update the path as needed)
model = load_model('./dataset/model/Model-6-Size-[150,150,3]-Conv[4]-Batch[64]-LR[0.001]-Epoch[50]-Plant Disease Sequential-98.13.h5')

plant_names = [
    "Apple Cedar_apple_rust",
    "Apple Healty",
    "Grape Blackrot",
    "Grape healthy",
    "Peach Bacterial_spot",
    "Peach healthy",
    "Potato Late blight",
    "Potato healthy",
    "Straweberry Leaf_scorch",
    "Straweberry healthy",
    "Tomato Bacterial spot",
    "Tomato healthy",
    
]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Update target_size sesuai kebutuhan model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi jika diperlukan

    prediction = model.predict(img_array)
    # Lakukan decoding hasil prediksi jika diperlukan
    # Misalnya, jika model Anda mengklasifikasikan 3 kelas
    predicted_class_index = np.argmax(prediction)
    predicted_class = plant_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return predicted_class, confidence
