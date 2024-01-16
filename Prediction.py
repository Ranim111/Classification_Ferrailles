import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Charger le modèle entraîné à partir du fichier (remplacez 'modele_entrene.h5' par votre chemin vers le modèle)
model = load_model('modele_entrene.keras')

# Définir les classes pour les prédictions
classes = ["Billette","Fil","Khorda", "Rond_Beton"]

# Exemple de prédiction pour une image individuelle
image_path = 'fer.jpg'  # Remplacez par le chemin de votre image

# Charger l'image et la prétraiter pour l'entrée du modèle
img_height, img_width = 150, 150  # Taille souhaitée pour vos images
img = load_img(image_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalisez les valeurs des pixels (facultatif)

# Effectuer la prédiction sur l'image
predictions = model.predict(img_array)

# Le résultat de 'predictions' sera un tableau de probabilités pour chaque classe
# Vous pouvez choisir la classe avec la probabilité la plus élevée comme prédiction finale
predicted_class_index = np.argmax(predictions[0])
predicted_class_label = classes[predicted_class_index]

print("Prédiction :", predicted_class_label)
