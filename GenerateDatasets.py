from emnist import extract_training_samples
from emnist import extract_test_samples
import cv2
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import random
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import sys


def getMathDatasetAsNP(train_set,classes):
    math_X_dataset = []
    math_y_dataset = []
    for i in range(len(train_set)):
        img = train_set[i]['features'].reshape(24, 24)
        img = np.where(img == 1, 0, 255)
        img = np.pad(img, 2, mode='constant')
        math_X_dataset.append(img)
        idx = train_set[i]['label'].argmax()
        math_y_dataset.append(classes[idx])

    math_X_dataset = np.array(math_X_dataset)
    math_y_dataset = np.array(math_y_dataset)
    return math_X_dataset,math_y_dataset

def threshold(set):
    arr = []
    for i in range(set.shape[0]):
        img = set[i]
        img = cv2.convertScaleAbs(img)
        ret1, char = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr.append(char)

    return np.array(arr)


def concat_2_imgs(image1, image2):
    (cy, cx) = np.where(image1 == 255)
    (bottomy, bottomx) = (np.max(cy), np.max(cx))

    (cy, cx) = np.where(image2 == 255)
    (topy2, topx2) = (np.min(cy), np.min(cx))
    (bottomy2, bottomx2) = (np.max(cy), np.max(cx))
    r = random.randint(0, 3)

    new_img = np.zeros((28,56),dtype=np.uint8)
    new_img[0:image1.shape[0],0:image1.shape[1]] = image1
    if (bottomx-r)+(bottomx2-topx2)+1 > 56:
        raise Exception('Array index out of bound')
    new_img[:image2.shape[0], bottomx-r:(bottomx-r)+(bottomx2-topx2)+1] += image2[:,topx2:bottomx2+1]
    new_img = create_1_img(new_img)
    return new_img

def create_1_img(image1):
    (cy, cx) = np.where(image1 == 255)
    (topy, topx) = (np.min(cy), np.min(cx))
    (bottomy, bottomx) = (np.max(cy), np.max(cx))
    if topx == 0:
        topx = 1
    if topy == 0:
        topy = 1
    char = image1[0:28, topx - 1:bottomx + 2]
    h, w = char.shape[:2]
    s = int((56-w)/2)

    new_img = np.zeros((28, 56), dtype=np.uint8)
    new_img[0:char.shape[0], s:s+char.shape[1]] = char
    return new_img

def generate_connected_character_recognition_dataset():
    print("Generating connected character recognition datatset!")
    X_train, y_train = extract_training_samples('byclass')
    X_test, y_test = extract_test_samples('byclass')

    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    with open('./mathsymbols/train/train.pickle', 'rb') as train:
        print('Restoring math training set ...')
        math_train_set = pickle.load(train)

    with open('./mathsymbols/test/test.pickle', 'rb') as test:
        print('Restoring math test set ...')
        math_test_set = pickle.load(test)

    classes = open('./mathsymbols/classes.txt', 'r').read().split()
    math_X_train, math_y_train = getMathDatasetAsNP(math_train_set, classes)
    math_X_test, math_y_test = getMathDatasetAsNP(math_test_set, classes)

    math_X_train = np.concatenate((math_X_train, math_X_test), axis=0)
    math_y_train = np.concatenate((math_y_train, math_y_test), axis=0)

    math_y_train = math_y_train.astype(int)
    math_symbols = (math_y_train > 61)

    math_symbols_X = math_X_train[math_symbols]
    math_symbols_y = math_y_train[math_symbols]
    math_symbols_X = math_symbols_X.reshape(-1, 28, 28, 1)

    unique, counts = np.unique(math_symbols_y, return_counts=True)

    print(dict(zip(unique, counts)))

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.3,  # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)

    batch_size = 256
    aug_epochs = 12 * int(math_symbols_X.shape[0] / batch_size)
    batches = datagen.flow(math_symbols_X, math_symbols_y, batch_size=batch_size)
    aug_math_symbols_X = []
    aug_math_symbols_y = []
    for i in range(aug_epochs):
        batch = next(batches)
        batch_x = batch[0].reshape(-1, 28, 28)
        batch_y = batch[1]
        for k in range(batch_x.shape[0]):
            sym = batch_x[k]
            sym = sym.astype(int)
            aug_math_symbols_X.append(sym)
            aug_math_symbols_y.append(batch_y[k])

    math_X_train = np.array(aug_math_symbols_X)
    math_y_train = np.array(aug_math_symbols_y)

    X_train = np.array(X_train)

    X_train = np.concatenate((X_train, math_X_train), axis=0)
    y_train = np.concatenate((y_train, math_y_train), axis=0)
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    dct = dict(zip(unique, counts))
    min_class_amt = dct[min(dct, key=dct.get)]

    resampled_X = []
    resampled_y = []

    for i in range(72):
        indices = np.where(y_train == i)
        random_indices = np.random.choice(indices[0], min_class_amt, replace=False)
        res_X = X_train[random_indices]
        res_y = y_train[random_indices]
        resampled_X.extend(res_X)
        resampled_y.extend(res_y)

    X_train = np.array(resampled_X)
    y_train = np.array(resampled_y)

    print("After resampling...........")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    X_train, y_train = shuffle(X_train, y_train)

    X_train = threshold(X_train)

    X_train, y_train = shuffle(X_train, y_train)

    generated_X = []
    generated_y = []
    exxception_cnt = 0
    print(X_train.shape[0])

    for i in range(X_train.shape[0]):
        index = np.random.choice(X_train.shape[0], 2, replace=False)
        char1 = X_train[index[0]]
        char2 = X_train[index[1]]
        try:
            touch_img = concat_2_imgs(char1, char2)
        except Exception as e:
            # cv2.imwrite("./exceptionImgs/img1_" + str(exxception_cnt) + ".png", char1)
            # cv2.imwrite("./exceptionImgs/img2_" + str(exxception_cnt) + ".png", char2)
            exxception_cnt += 1
            continue;
        generated_X.append(touch_img)
        generated_y.append(1)
        if i % 10000 == 0:
            print("Generated: "+str(i))
        index = np.random.choice(X_train.shape[0], 1, replace=False)
        single_img = create_1_img(X_train[index[0]])
        generated_X.append(single_img)
        generated_y.append(0)
    print("Finished generating : " + str(i) + " images")
    generated_X = np.array(generated_X)
    generated_y = np.array(generated_y)

    generated_X, generated_y = shuffle(generated_X, generated_y)
    dataset = (generated_X, generated_y)

    pickle.dump(dataset, open("./Datasets/connectedCharacterRecognition.pickle", "wb"), protocol=4)
    no_of_samples = 1000
    index = np.random.choice(generated_X.shape[0], no_of_samples, replace=False)
    for i in range(no_of_samples):
        char1 = generated_X[index[i]]
        y = generated_y[index[i]]
        cv2.imwrite("./Datasets/connectedCharacterRecognition/samples/sample_touch" + str(i) + " y=" + str(y) + ".png", char1)
    print("Saved generated dataset!")


def generate_implicit_segmentation_dataset():
    print("Generating implicit segmentation dataset!")
    X_train, y_train = extract_training_samples('byclass')
    X_test, y_test = extract_test_samples('byclass')

    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    with open('./mathsymbols/train/train.pickle', 'rb') as train:
        print('Restoring math training set ...')
        math_train_set = pickle.load(train)

    with open('./mathsymbols/test/test.pickle', 'rb') as test:
        print('Restoring math test set ...')
        math_test_set = pickle.load(test)

    classes = open('./mathsymbols/classes.txt', 'r').read().split()
    math_X_train, math_y_train = getMathDatasetAsNP(math_train_set, classes)
    math_X_test, math_y_test = getMathDatasetAsNP(math_test_set, classes)

    math_X_train = np.concatenate((math_X_train, math_X_test), axis=0)
    math_y_train = np.concatenate((math_y_train, math_y_test), axis=0)

    math_y_train = math_y_train.astype(int)
    math_symbols = (math_y_train > 61)

    math_symbols_X = math_X_train[math_symbols]
    math_symbols_y = math_y_train[math_symbols]
    math_symbols_X = math_symbols_X.reshape(-1, 28, 28, 1)

    unique, counts = np.unique(math_symbols_y, return_counts=True)

    print(dict(zip(unique, counts)))

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.3,  # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)

    batch_size = 256
    aug_epochs = 12 * int(math_symbols_X.shape[0] / batch_size)
    batches = datagen.flow(math_symbols_X, math_symbols_y, batch_size=batch_size)
    aug_math_symbols_X = []
    aug_math_symbols_y = []
    for i in range(aug_epochs):
        batch = next(batches)
        batch_x = batch[0].reshape(-1, 28, 28)
        batch_y = batch[1]
        for k in range(batch_x.shape[0]):
            sym = batch_x[k]
            sym = sym.astype(int)
            aug_math_symbols_X.append(sym)
            aug_math_symbols_y.append(batch_y[k])

    math_X_train = np.array(aug_math_symbols_X)
    math_y_train = np.array(aug_math_symbols_y)

    X_train = np.array(X_train)

    X_train = np.concatenate((X_train, math_X_train), axis=0)
    y_train = np.concatenate((y_train, math_y_train), axis=0)
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    dct = dict(zip(unique, counts))
    min_class_amt = dct[min(dct, key=dct.get)]

    resampled_X = []
    resampled_y = []
    for i in range(72):
        indices = np.where(y_train == i)
        random_indices = np.random.choice(indices[0], min_class_amt, replace=False)
        res_X = X_train[random_indices]
        res_y = y_train[random_indices]
        resampled_X.extend(res_X)
        resampled_y.extend(res_y)

    X_train = np.array(resampled_X)
    y_train = np.array(resampled_y)

    print("After resampling...........")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    X_train, y_train = shuffle(X_train, y_train)

    X_train = threshold(X_train)

    X_train, y_train = shuffle(X_train, y_train)

    generated_X = []
    generated_y = []
    exxception_cnt = 0
    print(X_train.shape[0])


    for i in range(X_train.shape[0] * 20):
        if i % 1000 == 0:
            print(i)
        index = np.random.choice(X_train.shape[0], 2, replace=False)
        char1 = X_train[index[0]]
        char2 = X_train[index[1]]
        try:
            touch_img = concat_2_imgs(char1, char2)
        except Exception as e:
            # cv2.imwrite("./exceptionImgs/img1_" + str(exxception_cnt) + ".png", char1)
            # cv2.imwrite("./exceptionImgs/img2_" + str(exxception_cnt) + ".png", char2)
            exxception_cnt += 1
            continue;

        generated_X.append(touch_img)
        generated_y.append([int(y_train[index[0]]), int(y_train[index[1]])])
        if i % 72 == 0:
            index = np.random.choice(X_train.shape[0], 1, replace=False)
            single_img = create_1_img(X_train[index[0]])

            generated_X.append(single_img)
            generated_y.append([int(y_train[index[0]])])
    print("Finished generating : " + str(i) + " images")
    generated_X = np.array(generated_X)
    generated_y = np.array(generated_y)

    generated_X, generated_y = shuffle(generated_X, generated_y)
    dataset = (generated_X, generated_y)

    pickle.dump(dataset, open("./Datasets/implicit_segmentation_dataset.pickle", "wb"),
                protocol=4)
    no_of_samples = 1000
    index = np.random.choice(generated_X.shape[0], no_of_samples, replace=False)
    for i in range(no_of_samples):
        char1 = generated_X[index[i]]
        y = generated_y[index[i]]
        cv2.imwrite("./Datasets/implicitSegmentation/samples/sample_touch" + str(i) + " y=" + str(y) + ".png", char1)
    print("Saved generated dataset!")

def generate_single_character_dataset():
    print("Generating single character dataset!")
    X_train, y_train = extract_training_samples('byclass')
    X_test, y_test = extract_test_samples('byclass')

    with open('./mathsymbols/train/train.pickle', 'rb') as train:
        print('Restoring math training set ...')
        math_train_set = pickle.load(train)

    with open('./mathsymbols/test/test.pickle', 'rb') as test:
        print('Restoring math test set ...')
        math_test_set = pickle.load(test)

    classes = open('./mathsymbols/classes.txt', 'r').read().split()

    math_X_train, math_y_train = getMathDatasetAsNP(math_train_set,classes)
    math_X_test, math_y_test = getMathDatasetAsNP(math_test_set,classes)

    math_y_train = math_y_train.astype(int)
    math_symbols = (math_y_train>61) & (math_y_train!=70)
    math_symbols_other = (math_y_train<62) | (math_y_train==70)

    math_symbols_X = math_X_train[math_symbols]
    math_symbols_other_X = math_X_train[math_symbols_other]
    math_symbols_y = math_y_train[math_symbols]
    math_symbols_other_y =  math_y_train[math_symbols_other]
    math_symbols_X = math_symbols_X.reshape(-1,28,28,1)

    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.2, # Randomly zoom image
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)

    batch_size = 256
    aug_epochs = 5 * int(math_symbols_X.shape[0]/batch_size)
    batches = datagen.flow(math_symbols_X,math_symbols_y, batch_size=batch_size)
    aug_math_symbols_X = []
    aug_math_symbols_y = []
    for i in range(aug_epochs):
        batch = next(batches)
        batch_x =  batch[0].reshape(-1,28,28)
        batch_y =  batch[1]
        for k in range(batch_x.shape[0]):
            sym = batch_x[k]
            sym = sym.astype(int)
            aug_math_symbols_X.append(sym)
            aug_math_symbols_y.append(batch_y[k])

    aug_math_symbols_X = np.array(aug_math_symbols_X)
    aug_math_symbols_y = np.array(aug_math_symbols_y)

    aug_math_symbols_X_train, aug_math_symbols_X_test, aug_math_symbols_y_train, aug_math_symbols_y_test = train_test_split(aug_math_symbols_X,aug_math_symbols_y,test_size=0.1)

    math_X_train = np.concatenate((aug_math_symbols_X_train,math_symbols_other_X),axis=0)
    math_y_train = np.concatenate((aug_math_symbols_y_train,math_symbols_other_y),axis=0)
    math_X_test = np.concatenate((aug_math_symbols_X_test,math_X_test),axis=0)
    math_y_test = np.concatenate((aug_math_symbols_y_test,math_y_test),axis=0)

    X_train = np.concatenate((X_train,math_X_train),axis=0)
    y_train = np.concatenate((y_train,math_y_train),axis=0)

    X_test = np.concatenate((X_test,math_X_test),axis=0)
    y_test = np.concatenate((y_test,math_y_test),axis=0)

    X_train = threshold(X_train)
    X_test = threshold(X_test)

    X_train, y_train = shuffle(X_train,y_train)
    X_test, y_test = shuffle(X_test,y_test)
    dataset = (X_train, X_test, y_train, y_test)
    pickle.dump(dataset, open("./Datasets/single_character.pickle", "wb"),  protocol=4)
    print("Generated " + str(X_train.shape[0]) + " train images")
    print("Generated " + str(X_test.shape[0]) + " test images")
    no_of_samples = 1000
    index = np.random.choice(X_train.shape[0], no_of_samples, replace=False)
    for i in range(no_of_samples):
        char1 = X_train[index[i]]
        y = y_train[index[i]]
        cv2.imwrite("./Datasets/singleCharacter/samples/sample_" + str(i) + " y=" + str(y) +".png", char1)
    print("Saved generated dataset!")

def getEncoding_dict():
    dict = {}
    count = 0
    for i in range(72):
        for j in range(73):
            dict[(i,j)] = count
            count += 1
    return dict


def saveDatsetAsClasses():
    print("Saving implicit segmentation datatset as classes")
    encode_dict = getEncoding_dict()
    with open("./Datasets/implicit_segmentation_dataset.pickle", "rb") as f:
        X,y = pickle.load(f)
    X = np.array(X)
    y = np.array(y)
    y = sequence.pad_sequences(y, padding='post', value=72)
    encoded_y = []
    for i in range(len(y)):
        yi = y[i]
        k = yi[0]
        l = yi[1]
        encoded_y.append(encode_dict[(k, l)])

    y = np.array(encoded_y)
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    X = []
    y = []
    import gc
    gc .collect()
    import os

    os.mkdir("./Datasets/implicitSegmentation/train")
    os.mkdir("./Datasets/implicitSegmentation/test")

    img_count_train_dict = {}
    img_count_test_dict = {}
    for k1,k2 in encode_dict:
        value = encode_dict[(k1,k2)]
        img_count_train_dict[value] = 0
        img_count_test_dict[value] = 0
        os.mkdir("./Datasets/implicitSegmentation/test/class_"+str(value))
        os.mkdir("./Datasets/implicitSegmentation/train/class_"+str(value))

    for i in range(len(y_train)):
        img = X_train[i]
        key = y_train[i]
        img_number = img_count_train_dict[key]
        cv2.imwrite("./Datasets/implicitSegmentation/train/class_"+str(key)+"/img_"+str(img_number)+".jpg",img)
        img_count_train_dict[key] = img_number + 1
    X_train = []
    y_train = []
    gc.collect()
    for i in range(len(y_test)):
        img = X_test[i]
        key = y_test[i]
        img_number = img_count_test_dict[key]
        cv2.imwrite("./Datasets/implicitSegmentation/test/class_" + str(key) + "/img_" + str(img_number) + ".jpg",img)
        img_count_test_dict[key] = img_number + 1

    print("Saved dataset as classes!")

if sys.argv[1] == 'connectedCharacterRecognition':
    generate_connected_character_recognition_dataset()
elif sys.argv[1] == 'implicitSegmentation':
    generate_implicit_segmentation_dataset()
    saveDatsetAsClasses()
elif sys.argv[1] == 'singleCharacter':
    generate_single_character_dataset()
else:
    print("Wrong argument!")