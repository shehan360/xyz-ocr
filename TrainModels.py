from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D, Activation, Dense, BatchNormalization
from keras.utils import np_utils
from keras.models import save_model
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import math
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
import sys

def train_connected_character_recognition_model():
    with open("./Datasets/connectedCharacterRecognition.pickle", "rb") as f:
        X, y = pickle.load(f)
    y = np.array(y)
    X = np.array(X)
    X = X.astype('float32')
    X = X / 255.0
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    X = X.reshape((-1, 28, 56, 1))
    y = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    print('Number of images in x_train', X_train.shape[0])
    print('Number of images in x_test', X_test.shape[0])

    input_shape = (28, 56, 1)
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(units=2, activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    batch_size = 128
    steps = math.ceil(X_train.shape[0] / batch_size)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.15,  # Randomly zoom image
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                  epochs=10, validation_data=(X_test, y_test),
                                  verbose=2, steps_per_epoch=steps)

    save_model(model, './Models/connected_character_recognition_8cnn.h5')
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print(score)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

def train_implicit_segmentation_model():
    model = Sequential()
    mod = ResNet50(include_top=False, weights='imagenet', input_shape=(56, 112, 3), pooling='max')
    mod.summary()
    model.add(ResNet50(include_top=False, weights='imagenet', input_shape=(56, 112, 3), pooling='max'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(units=5256, activation='softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    batch_size = 64
    epochs = 20
    train_steps = math.ceil(2907882 / batch_size)
    valid_steps = math.ceil(323098 / batch_size)

    checkpoint = ModelCheckpoint("./Models/implicit_segmentation_model.hdf5",
                                 monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.15,
        width_shift_range=0.15)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        './Datasets/implicitSegmentation/train',
        target_size=(56, 112),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb')

    validation_generator = test_datagen.flow_from_directory(
        './Datasets/implicitSegmentation/test',
        target_size=(56, 112),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        verbose=2,
        validation_data=validation_generator,
        validation_steps=valid_steps, callbacks=[checkpoint])
    save_model(model, './Models/implicit_segmentation_model.h5')
    batch_size = 146
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        './Datasets/implicitSegmentation/test',
        target_size=(56, 112),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb')
    valid_steps = math.ceil(323098 / batch_size)
    score = model.evaluate_generator(validation_generator, steps=valid_steps, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print(score)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

def train_single_character_model():
    with open("./Datasets/single_character.pickle", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    X_train = X_train.astype('float16')
    X_test = X_test.astype('float16')

    X_train = np.array(X_train) / 255.0
    y_train = np.array(y_train)
    X_test = np.array(X_test) / 255.0
    y_test = np.array(y_test)

    print('x_train shape:', X_train.shape)
    print('Number of images in x_train', X_train.shape[0])
    print('Number of images in x_test', X_test.shape[0])

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)


    y_train = y_train.astype(int)
    bc = np.bincount(y_train)
    print(bc)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(class_weights))
    y_train = np_utils.to_categorical(y_train, 72)
    y_test = np_utils.to_categorical(y_test, 72)

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False, input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.3))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.3))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(units=72, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)
    checkpoint = ModelCheckpoint("./Models/single_char_model_6cnn.hdf5",
                                 monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)

    batch_size = 64
    steps = math.ceil(X_train.shape[0] / batch_size)
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                  epochs=15, validation_data=(X_test, y_test),
                                  verbose=2, steps_per_epoch=steps, callbacks=[checkpoint])

    save_model(model, './Models/single_char_model_6cnn.h5')

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print(score)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

if sys.argv[1] == 'connectedCharacterRecognition':
    train_connected_character_recognition_model()
elif sys.argv[1] == 'implicitSegmentation':
    train_implicit_segmentation_model()
elif sys.argv[1] == 'singleCharacter':
    train_single_character_model()
else:
    print("Wrong argument!")