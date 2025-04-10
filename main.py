import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

# print("TensorFlow version:", tf.__version__)

# Charger et préparer les données
## Images
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

## Résultats
nb_epochs = 10 # valeur initiale = 10
lr = 0.003 # Learning rate : valeur par défaut = 0.001
nb_conv = 2 # Nombre de couches convolutionnelles (intialement 2)
kernel_size = 3 # Taille du noyau de convolution (intitialement 3)

res_dir = './results'
res_name = f"{nb_epochs}_{lr}_{nb_conv}_{kernel_size}"

# Définition du callback TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Définir le modèle CNN
model = Sequential([
  Conv2D(32, (kernel_size, kernel_size), activation='relu', input_shape=(64, 64, 3)),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(64, (kernel_size, kernel_size), activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(1, activation='sigmoid') ])

# Compilation du modèle
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(
  train_generator, 
  validation_data=val_generator, 
  epochs=nb_epochs,
  callbacks=[tensorboard_callback])

# Évaluation
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')

with open(res_dir+"/"+"eval"+"-"+res_name+"-report.txt",'w',encoding="utf-8") as f:
  f.write(f'Loss: {loss}, Accuracy: {accuracy}' + '\n')
  f.write(f'Model summary : ' + '\n')
  model.summary(print_fn=lambda x: f.write(x + '\n'))