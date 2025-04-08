import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

# print("TensorFlow version:", tf.__version__)

# Charger et préparer les données
train_dir = './dataset/train'
val_dir = './dataset/test'
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
  train_dir, target_size=(64, 64),
  batch_size=32,
  class_mode='binary')

val_generator = datagen.flow_from_directory(
  val_dir,
  target_size=(64, 64),
  batch_size=32,
  class_mode='binary')

# Définition du callback TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Définir le modèle CNN
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(1, activation='sigmoid') ])

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(
  train_generator, 
  validation_data=val_generator, 
  epochs=3,
  callbacks=[tensorboard_callback])

# Évaluation
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')