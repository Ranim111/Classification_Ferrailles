import os
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
import scipy
# 1. Définir les chemins vers les dossiers de données
train_data_dir = r'C:\Users\ranim\AppData\Local\Programs\Python\Classification\Entrainement'
test_data_dir = r'C:\Users\ranim\AppData\Local\Programs\Python\Classification\Test'

# 2. Prétraiter les images et charger les données
input_shape = (150, 150)  # Définissez la taille souhaitée pour vos images

# Utilisez ImageDataGenerator pour charger les images depuis les chemins et les prétraiter automatiquement
train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape,
    batch_size=32,
    class_mode='categorical'  # Pour la classification multiclasse
)

test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape,
    batch_size=32,
    class_mode='categorical'
)

# 3. Construire l'architecture du modèle
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  
])

# 4. Compiler le modèle
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 5. Entraîner le modèle
model.fit(train_generator,
          epochs=10,  # Choisissez le nombre d'époques approprié
          steps_per_epoch=len(train_generator),
          validation_data=test_generator,
          validation_steps=len(test_generator))

# 6. Évaluer les performances du modèle
loss, accuracy = model.evaluate(test_generator)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
# Supposons que vous avez déjà entraîné votre modèle et qu'il est stocké dans la variable 'model'

# Chemin vers le dossier où vous souhaitez sauvegarder votre modèle
# model_save_path = r'C:\Users\bejao\AppData\Local\Programs\Python\Python311\model'

# Utilisation de la méthode 'save' pour sauvegarder le modèle

# 6. Sauvegarder le modèle au format .keras
model.save('modele_entrene.keras')
