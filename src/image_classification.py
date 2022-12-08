import os
import cv2
import imghdr
import shutil
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import fiftyone as fo
import fiftyone.zoo as foz
import tensorflow_datasets as tfds
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from tensorflow.python.keras.metrics import Precision, Recall, BinaryAccuracy, SparseCategoricalAccuracy


data_dir = "src\\data"
model_dir = "src\\models"
image_exts = ["jpeg", "jpg", "bmp"]

def train():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    cpus = tf.config.experimental.list_physical_devices("CPU")

    #Avoid Out-of-Memory errors by setting GPU Memory Growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(gpus)

    #Dataset
    data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=50, shuffle=True, image_size=(224,224))
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    plt.show()


    #Preprocessing
    data = data.map(lambda x,y: (x/255, y))
    scaled_iterator = data.as_numpy_iterator()

    #Creating train/test/val partitions
    train_size = int(len(data) * .7)
    val_size = int(len(data)*.2)
    test_size = len(data) - train_size - val_size

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)


    #NETWORK ARCHITECTURE
    model = Sequential()
    model.add(Conv2D(96, (3,3), 1, activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    
    #model.add(Conv2D(16, (3,3), 1, activation='relu'))
    #model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()


    #TRAIN

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    EPOCHS = 50

    hist = model.fit(train, epochs=EPOCHS, validation_data=val, callbacks=[tensorboard_callback])

    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc='upper left')
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc='upper left')
    plt.show()

    #Evaluation
    """
    accuracy = SparseCategoricalAccuracy()
    
    for batch in test.as_numpy_iterator():
        x, y = batch
        print(y.size)
        yhat = model.predict(x)
        yhat = list(map(lambda x: np.argmax(x), yhat))
        yhat = np.asarray(yhat).reshape((1,y.size))
        accuracy.update_state(y, yhat)
        
    print(f'Accuracy: {accuracy.result().numpy()}')
    """
    #Test
    
    img = cv2.imread("frames\\frame_1.jpg")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    resize = tf.image.resize(img, (224,224))
    plt.imshow(resize.numpy().astype(int))
    
    np.expand_dims(resize, 0)
    yhat = model.predict(np.expand_dims(resize/255, 0))
    print(yhat)
    
    now = datetime.now()
    time_stamp = now.strftime("%H.%M_%d-%m-%y")
    model_path = os.path.join(model_dir, time_stamp)
    os.mkdir(model_path)
    model.save(os.path.join(model_path, "model_{}.h5".format(time_stamp)))
    model.save_weights(os.path.join(model_path, "weights_{}".format(time_stamp)))

    plt.show()
    
def clean_dataset():
    for image_class in os.listdir(data_dir):
        print(image_class)
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print(e)
                print('Error with image {}'.format(image_path))
                exit()


def test_train():
    
    dataset_name = "open-images-v6"
    dataset = foz.load_zoo_dataset(
        dataset_name,
        split="validation",
        label_types=["detections","classifications"],
        classes=["Person"],
        shuffle=True,
        max_samples=100)
    
    session = fo.launch_app(dataset, desktop=True)
    session.wait()


if __name__ == '__main__':
    train()