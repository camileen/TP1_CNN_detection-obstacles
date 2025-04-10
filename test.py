import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

# print("TensorFlow version:", tf.__version__)

def test(epochs=10, lr=0.001, nb_conv=2, kernel_size=3, name="") :
  # Charger et préparer les données
  ## Images
  train_dir = './dataset/train'
  val_dir = './dataset/test'
  datagen = ImageDataGenerator(rescale=1./255)

  train_generator = datagen.flow_from_directory(
    train_dir, target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

  val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

  ## Résultats
  res_dir = './results'
  res_name = f"{epochs}_{lr}_{nb_conv}_{kernel_size}"

  # Définition du callback TensorBoard
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  # Définir le modèle CNN
  model = Sequential()
  model.add(Conv2D(32, (kernel_size, kernel_size), activation='relu', input_shape=(64, 64, 3)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if (nb_conv > 1):
    for i in range(1, nb_conv):
        model.add(Conv2D(32*(2**i), (kernel_size, kernel_size), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(10, activation='softmax'))

  # Compilation du modèle
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

  # Entraînement du modèle
  model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=epochs,
    callbacks=[tensorboard_callback])

  # Évaluation
  loss, accuracy = model.evaluate(val_generator)
  print(f'Loss: {loss}, Accuracy: {accuracy}')

  with open(res_dir+"/"+name+"-"+res_name+"-report.txt",'w',encoding="utf-8") as f:
    f.write(f'Loss: {loss}, Accuracy: {accuracy}' + '\n')
    f.write(f'Model summary : ' + '\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
  

if __name__ == "__main__":
  list_lr = [0.0008, 0.0005]
  # list_ker_size = [3, 5, 7]
  # list_nb_conv = [2, 3, 4]

  for lr in list_lr:
    test(lr=lr, name="lr")

  # for kernel_size in list_ker_size:
  #   test(kernel_size=kernel_size, name="kernel_size")

  # for nb_conv in list_nb_conv:
  #   test(nb_conv=nb_conv, name="nb_conv")
