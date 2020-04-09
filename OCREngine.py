import cv2
import numpy as np
import pickle
import os
import glob
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage import img_as_ubyte
import random

def getConnectingPoints(topPoints,btmPoints,h):
    points = []
    if len(topPoints) == 1 and len(btmPoints) == 1:
        points.append([topPoints[0],btmPoints[0]])
        return points

    if len(btmPoints) == 0 and len(topPoints) != 0:
        sorted_toppoints = sorted(topPoints,key=lambda x: x[1], reverse= True)
        btmPoints.append([sorted_toppoints[0][0], h - 1])
        points.append([sorted_toppoints[0], btmPoints[0]])
        return points

    if len(btmPoints) == 1 and len(topPoints) == 0:
        points.append([[btmPoints[0][0], 1], btmPoints[0]])
        return points

    if len(btmPoints) == 1 and len(topPoints) > 1:
        distances = []
        for topPoint in topPoints:
            y = abs(topPoint[1]-btmPoints[0][1])
            x = abs(topPoint[0] - btmPoints[0][0])
            distances.append(x+y)
        min_idx = min(range(len(distances)), key=distances.__getitem__)
        points.append([topPoints[min_idx],btmPoints[0]])
        return points

    return points

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def splitCharsUsingContours(skel,char,new_line,charKey):
    contours, hierarchy = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    parent0conts = []
    for i in range(len(contours)):
        if hierarchy[i][3] == 0:
            parent0conts.append(i)
    if len(parent0conts) > 2:
        cont_areas = []
        for i in parent0conts:
            contArea = cv2.contourArea(contours[i])
            cont_areas.append(contArea)
        maxA = max(cont_areas)
        maxInd1 = cont_areas.index(maxA)
        cont_areas.remove(maxA)
        maxB = max(cont_areas)
        maxInd2 = cont_areas.index(maxB)
        parent0conts = [maxInd1,maxInd2]

    # backtorgb = cv2.cvtColor(char, cv2.COLOR_GRAY2RGB)
    # color = random_color()
    # cv2.drawContours(backtorgb, contours, -1, (0,0,255), thickness=1)
    # cv2.imwrite("./reportImgs/rgb" + str(0) + "_" + str(charKey) + ".png", backtorgb)
    # if charKey == 263:
    #     print()
    #
    # for i in parent0conts:
    #     backtorgb = cv2.cvtColor(char, cv2.COLOR_GRAY2RGB)
    #     color = random_color()
    #     cv2.drawContours(backtorgb, contours, i, color, thickness=1)
    #     cv2.imwrite("./reportImgs/rgb"+ str(i)+"_" + str(charKey) + ".png", backtorgb)

    if len(parent0conts) == 2:

        for i in parent0conts:
            mask1 = np.zeros_like(char)
            cv2.drawContours(mask1, contours, i, 255, -1)
            out = np.zeros_like(char)  # Extract out the object and place into output image
            out[mask1 == 255] = char[mask1 == 255]
            (y, x) = np.where(out == 255)
            try:
                (topy, topx) = (np.min(y), np.min(x))
                (bottomy, bottomx) = (np.max(y), np.max(x))
            except Exception as e:
                continue
            char1 = out[topy-1:bottomy + 2, topx-1:bottomx + 2]
            newKey = charKey+topx
            new_line[newKey] = {}
            new_line[newKey]['char'] = char1
            new_line[newKey]['prediction'] = -1
            new_line[newKey]['type'] = "single"
            new_line[newKey]['touch_classification'] = 0
            # cv2.imshow("ovrlp" + str(lineKey)+str(index)+ str(i), char1)
        return True,contours

    return False, contours

def create_1_img_v2(tw,th,image1):

    h, w = image1.shape[:2]

    if w<=tw and h<=th:
        sw = int((tw-w)/2)
        sh = int((th - h) / 2)
    else:
        if w>tw and h<=th:
            new_w = tw-1
            new_h = int((new_w)/(w/h))
        elif h>th and w<=tw:
            new_h = th-1
            new_w = int((new_h)*(w/h))
        else:
            if w > h:
                new_w = tw - 1
                new_h = int((new_w) / (w / h))
                if new_h > (th-1):
                    new_h = th - 1
                    new_w = int((new_h) * (w / h))
            elif w < h:
                new_h = th - 1
                new_w = int((new_h) * (w / h))
            else:
                new_h = th - 1
                new_w = tw - 1
        sw = int((tw - new_w)/2)
        sh = int((th - new_h) / 2)
        try:
            image1 = cv2.resize(image1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(e)
    new_img = np.zeros((th, tw), dtype=np.uint8)
    try:
        new_img[sh:sh + image1.shape[0], sw:sw + image1.shape[1]] = image1
    except Exception as e:
        print()
    #new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

    return new_img

def getPoints(image,kernel):
    ih,iw = image.shape
    kh,kw = kernel.shape
    k = int(kw/2)
    points = []
    for y in np.arange(k, ih-k):
        for x in np.arange(k, iw-k):
            roi = image[y - k:y + k + 1, x - k:x + k + 1]
            multip = (roi*kernel).sum()
            if multip > 3059:
                points.append([x,y])

    return points

def concat_2_imgs_h(image1, image2):
    new_width = image1.shape[1] + image2.shape[1]

    if image1.shape[0] > image2.shape[0]:
        new_height = image1.shape[0]
    else:
        new_height = image2.shape[0]

    new_img = np.zeros((new_height,new_width),dtype=np.uint8)
    new_img[0:image1.shape[0],0:image1.shape[1]] = image1
    new_img[:image2.shape[0], image1.shape[1]:image2.shape[1]+image1.shape[1]] = image2
    return new_img

def concat_to_near_line(lines,x,y,borderedline):
    linekeys = list(lines.keys())
    for linekey in linekeys:
        if abs(linekey[1] - y) <= 15:
            old_line = lines[linekey]
            new_line = concat_2_imgs_h(old_line, borderedline)
            lines[linekey] = new_line
            return True

    return False

def getEncoding_dict():
    dict = {}
    count = 0
    for i in range(72):
        for j in range(73):
            dict[(i,j)] = count
            count += 1
    return dict

def runOcrEngine(filename, touch_char_segementation_model, connected_character_recognition_model, single_char_model):
    files = glob.glob('./outputImgs/*')
    for f in files:
        os.remove(f)

    img = cv2.imread(filename)

    size = (1800, 2400)
    rimg = cv2.resize(img, size)
    gr = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./outputImgs/resized.png", gr)
    blur = cv2.GaussianBlur(gr, (13, 13), 0)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 15))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, element)
    # cv2.imshow('blackhat', blackhat)
    cv2.imwrite("./outputImgs/blackhat.png", blackhat)

    ret1, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #cv2.imshow('thresh', thresh)
    cv2.imwrite("./outputImgs/thresholded.png", thresh)
    inverted = cv2.bitwise_not(thresh)
    cv2.imwrite("./outputImgs/inverted.png", inverted)

    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element)
    # cv2.imshow('morphed', morphed)
    cv2.imwrite("./outputImgs/morphed.png", morphed)

    contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

    lines = {}
    vertical_projections = {}

    relevant_contours = {}
    for contour in contours:
        if (cv2.contourArea(contour) > 100):
            x, y, width, height = cv2.boundingRect(contour)
            relevant_contours[(x, y)] = contour

    for contourKey in sorted(relevant_contours):
        contour = relevant_contours[contourKey]
        x, y, width, height = cv2.boundingRect(contour)
        line = thresh[y:y + height, x:x + width]
        borderedline = cv2.copyMakeBorder(line, 1, 1, 5, 5, cv2.BORDER_CONSTANT, 0)

        can_concat = concat_to_near_line(lines, x, y, borderedline)
        if not can_concat:
            lines[(x, y)] = borderedline

    for lineKey in lines:
        line = lines[lineKey]
        # cv2.imshow("a" + str(lineKey), line)
        cv2.imwrite("./outputImgs/line" + str(lineKey) + ".png", line)
        proj = np.sum(line, axis=0).tolist()
        proj = np.array(proj)
        proj = proj / 255
        vertical_projections[lineKey[1]] = proj

    segmentation_columns = {}

    for projectionKey in vertical_projections:
        proj = vertical_projections[projectionKey]
        fs = list()
        for i in range(1, len(proj)):
            if proj[i] != 0:
                if proj[i - 1] == 0:
                    fs.append(i - 1)
            else:
                if proj[i - 1] != 0:
                    fs.append(i)
        segmentation_columns[projectionKey] = fs

    math_symbols = {}
    with open("./mathsymbols/classesSymbol.txt") as f:
        for line in f:
            (key, val) = line.split()
            math_symbols[int(key)] = val

    with open('D:/Dataset/class_indices.pickle', 'rb') as fp:
        class_indices = pickle.load(fp)
    key_list = list(class_indices.keys())
    val_list = list(class_indices.values())
    encoded_dict = getEncoding_dict()
    encoded_key_list = list(encoded_dict.keys())
    encoded_val_list = list(encoded_dict.values())
    line_chars = {}

    for index, (segement_colKey, lineKey) in enumerate(zip(segmentation_columns, lines)):
        line = lines[lineKey]
        (lx, ly) = lineKey
        segement_col = segmentation_columns[segement_colKey]
        height, width = line.shape
        chars = {}

        for i in range(0, len(segement_col), 2):
            contour = []
            char = {}
            contour.append([segement_col[i], 0])
            contour.append([segement_col[i], height - 1])
            contour.append([segement_col[i + 1], height - 1])
            contour.append([segement_col[i + 1], 0])
            contour = np.array(contour, dtype=np.int32)
            x, y, width, height = cv2.boundingRect(contour)
            char['img'] = line[y:y + height, x:x + width]
            char['type'] = "not classified"
            (cy, cx) = np.where(char['img'] == 255)
            (topy, topx) = (np.min(cy), np.min(cx))
            (bottomy, bottomx) = (np.max(cy), np.max(cx))
            char['img'] = char['img'][topy - 1:bottomy + 2, topx - 1:bottomx + 2]
            n_white_pix = np.sum(char['img'] == 255)

            resized_char = create_1_img_v2(112, 56, char['img'])
            #resized_char = unsharp_mask(resized_char)
            resized_char = cv2.cvtColor(resized_char, cv2.COLOR_GRAY2RGB)
            char['resized'] = resized_char
            resized_char = resized_char.astype('float32')
            resized_char = np.array(resized_char) / 255.0
            resized_char = resized_char.reshape(56, 112, 3)
            pred = touch_char_segementation_model.predict([[resized_char]])
            pred = pred.argmax()
            pred_class = key_list[val_list.index(pred)]
            pred_class_no = pred_class.split('_')
            pred_class_no = pred_class_no[1]

            seg_prediction = encoded_key_list[encoded_val_list.index(int(pred_class_no))]
            classified_seg_prediction = []
            for pred in seg_prediction:
                if pred >= 10 and pred <= 35:
                    pred = chr(pred + 55)
                elif pred > 35 and pred <= 61:
                    pred = chr(pred + 61)
                elif pred > 61 and pred < 72:
                    pred = math_symbols[pred]
                classified_seg_prediction.append(pred)
            char['seg_prediction'] = classified_seg_prediction

            resized_char = create_1_img_v2(112, 56, char['img'])
            resized_char = cv2.resize(resized_char, (56, 28), interpolation=cv2.INTER_AREA)
            resized_char = resized_char.astype('float32')
            resized_char = np.array(resized_char) / 255.0
            resized_char = resized_char.reshape(28, 56, 1)
            pred = connected_character_recognition_model.predict([[resized_char]])
            pred = pred.argmax()

            char['touch_classification'] = pred
            if n_white_pix <= 100:
                char['type'] = "single"
            chars[lx + x - 10] = char
        line_chars[ly] = chars


    implicit_seg_str = ''
    for lineKey in sorted(line_chars):
        line = line_chars[lineKey]
        for index, charKey in enumerate(sorted(line)):
            char = line[charKey]['resized']
            pred = line[charKey]['touch_classification']
            type = line[charKey]['type']
            seg_prediction = line[charKey]['seg_prediction']
            if seg_prediction[1] == 72:
                implicit_seg_str += str(seg_prediction[0]) + " "
            else:
                implicit_seg_str += str(seg_prediction[0]) + str(seg_prediction[1]) + " "
            # cv2.imshow("c" + str(lineKey) + " " + str(charKey) + " touch_classification = " + str(pred)+" type= "+str(type), char)
            cv2.imwrite(
                "./outputImgs/res" + str(lineKey) + " " + str(charKey) + " touch_pred = " + str(pred) + " type= " + str(
                    type) + " seg_pred = " + str(seg_prediction) + ".png", char)
        implicit_seg_str += "\n"

    for lineKey in line_chars:
        line = line_chars[lineKey]
        new_line = {}
        for index, charKey in enumerate(line):
            char = line[charKey]['img']
            if line[charKey]['touch_classification'] == 1:
                # cv2.imshow('long '+str(width), char)
                thres = threshold_otsu(char)
                binary = char > thres
                binary = invert(binary)
                skel = skeletonize(binary)
                skel = img_as_ubyte(skel)
                # cv2.imshow("SKELL", skel)
                splittable, contours = splitCharsUsingContours(skel, char, new_line, charKey)
                if splittable:
                    continue

                mask1 = np.zeros_like(char)
                cv2.drawContours(mask1, contours, 1, 255, 1)
                # cv2.imshow("mask1", mask1)
                h, w = mask1.shape
                w = int(w / 3)
                m = mask1[:, w:2 * w]
                # cv2.imshow("m", m)
                topkernel = np.array((
                    [2, 2, 2],
                    [-10, 10, -10],
                    [-10, -10, -10]), dtype="int")
                topkernel2 = np.array((
                    [-10, -10, -10],
                    [-10, 10, 2],
                    [-10, -10, -10]), dtype="int")
                btmkernel = np.array((
                    [-10, -10, -10],
                    [-10, 10, -10],
                    [2, 2, 2]), dtype="int")
                toppoints = getPoints(m, topkernel)
                toppoints2 = getPoints(m, topkernel2)
                if len(toppoints2) != 0:
                    toppoints.extend(toppoints2)
                btmpoints = getPoints(m, btmkernel)

                cnctpoints = getConnectingPoints(toppoints, btmpoints, h)

                if len(cnctpoints) == 1:
                    cv2.line(mask1, (cnctpoints[0][0][0] + w, cnctpoints[0][0][1]),
                             (cnctpoints[0][1][0] + w, cnctpoints[0][1][1]), (255, 255, 255), 1)
                # cv2.imshow("mask f", mask1)
                splittable, contours = splitCharsUsingContours(mask1, char, new_line, charKey)
                if splittable:
                    continue
                char_type = "multiple"
                touch_classification = 1
            else:
                char_type = line[charKey]['type']
                touch_classification = line[charKey]['touch_classification']
            new_line[charKey] = {}
            new_line[charKey]['char'] = char
            new_line[charKey]['prediction'] = -1
            new_line[charKey]['type'] = char_type
            new_line[charKey]['touch_classification'] = touch_classification
        line_chars[lineKey] = new_line

    for lineKey in sorted(line_chars):
        line = line_chars[lineKey]
        for charKey in sorted(line):
            char = line[charKey]['char']
            height, width = char.shape
            if height < 4 or width < 4:
                del line_chars[lineKey][charKey]
                continue

            if line[charKey]['touch_classification'] == 0:
                char = create_1_img_v2(56, 56, char)
                char = cv2.resize(char, (28, 28), interpolation=cv2.INTER_AREA)

                char = unsharp_mask(char)
                char = char.astype('float32')
                char = np.array(char) / 255.0
                char = char.reshape(28, 28, 1)
                pred = single_char_model.predict([[char]])
                pred = pred.argmax()
                if pred >= 10 and pred <= 35:
                    pred = chr(pred + 55)
                elif pred > 35 and pred <= 61:
                    pred = chr(pred + 61)
                elif pred > 61 and pred < 72:
                    pred = math_symbols[pred]
                line[charKey]['prediction'] = pred

    explicit_seg_string = ""
    for lineKey in sorted(line_chars):
        line = line_chars[lineKey]
        for charKey in sorted(line):
            if line[charKey]['prediction'] != -1:
                explicit_seg_string += str(line[charKey]['prediction']) + " "
            # else:
            #     explicit_seg_string += " $ "
        explicit_seg_string += "\n"

    return implicit_seg_str, explicit_seg_string

