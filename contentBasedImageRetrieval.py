import numpy as np
import os
import math
from PIL import Image


def histogram_rgb(im):
    height, width = im.size
    count_r = 0
    count_g = 0
    count_b = 0
    histogram_r = np.zeros(256)
    histogram_g = np.zeros(256)
    histogram_b = np.zeros(256)

    for x in range(0, height):
        for y in range(0, width):
            r, g, b = im.getpixel((x, y))
            histogram_r[r] += 1
            histogram_g[g] += 1
            histogram_b[b] += 1

    for i in range(0, 256):
        count_r += histogram_r[i]
        count_g += histogram_g[i]
        count_b += histogram_b[i]
    for i in range(0, 256):
        histogram_r[i] /= count_r
        histogram_g[i] /= count_g
        histogram_b[i] /= count_b

    return histogram_r, histogram_g, histogram_b


def histogram_hsv(im):
    height, width = im.size
    count = 0
    histogram = np.zeros(360)

    for x in range(0, height):
        for y in range(0, width):
            r, g, b = im.getpixel((x, y))
            h = rgb_to_hsv(r, g, b)
            histogram[int(h)] += 1

    for i in range(0, 360):
        count += histogram[i]
    for i in range(0, 360):
        histogram[i] /= count

    return histogram


def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360

    return h


def calculate_distance_rgb(histogramr1, histogramr2, histogramg1, histogramg2, histogramb1, histogramb2):
    distance = 0
    for x in range(0, 256):
        rd = histogramr1[x] - histogramr2[x]
        gd = histogramg1[x] - histogramg2[x]
        bd = histogramb1[x] - histogramb2[x]
        distance += math.sqrt(rd * rd + gd * gd + bd * bd)

    return distance


def calculate_distance_hsv(histogramh1, histogramh2):
    distance = 0
    for x in range(0, 360):
        d = histogramh1[x] - histogramh2[x]
        distance += math.sqrt(d * d)

    return distance


def take_images(tests, images):
    path = "D:/ytu/data/4/güz/Görüntü İşleme/ödevler/2/Content-Based-Image-Retrieval-master/data/camera/"
    folders = ["octopus", "elephant", "flamingo", "kangaroo", "Leopards", "sea_horse"]

    for i in folders:
        images, tests = take_from_dataset(images, tests, path + i + "/")
    print("Tüm dosyalar okundu.")

    return images, tests


def take_from_dataset(image, test, path):
    parent_list = os.listdir(path)
    for i in range(0, 20):
        filename = path + parent_list[i]
        image.append(filename)
    for i in range(20, 30):
        filename = path + parent_list[i]
        test.append(filename)

    return image, test


def save_histograms(images, rgb_histograms, hsv_histograms):
    for i in range(0, 120):
        image2 = Image.open(images[i]).convert('RGB')
        histogramr2, histogramg2, histogramb2 = histogram_rgb(image2)

        rgb_histograms.append(histogramr2)
        rgb_histograms.append(histogramg2)
        rgb_histograms.append(histogramb2)

        histogramh = histogram_hsv(image2)
        hsv_histograms.append(histogramh)

    return rgb_histograms, hsv_histograms


def find_similar_images(distances, images):
    for x in range(0, 5):
        i = distances.index(min(distances))
        print(images[i])
        distances[i] = 999999
    distances.clear()


def rgb_distance(train_images, test_images, rgb_histograms):
    print("~~RGB Uzayında En Benzer 10 Resim~~")

    for count in range(0, 60):
        distances = []
        image1 = Image.open(test_images[count]).convert('RGB')
        print([count+1], ". resim: ", test_images[count])
        histogramr1, histogramg1, histogramb1 = histogram_rgb(image1)

        j = 0
        while j < 360:
            histogramr2 = rgb_histograms[j]
            j += 1
            histogramg2 = rgb_histograms[j]
            j += 1
            histogramb2 = rgb_histograms[j]
            distance = calculate_distance_rgb(histogramr1, histogramr2, histogramg1, histogramg2, histogramb1, histogramb2)
            distances.append(distance)
            j += 1

        find_similar_images(distances, train_images)


def hue_distance(train_images, test_images, hsv_histograms):
    print("~~HSV Uzayında En Benzer 10 Resim~~")

    for count in range(0, 60):
        distances = []
        image1 = Image.open(test_images[count]).convert('RGB')
        print(count+1, ". resim: ", test_images[count])
        histogramh1 = histogram_hsv(image1)

        j = 0
        while j < 120:
            histogramh2 = hsv_histograms[j]
            distance = calculate_distance_hsv(histogramh1, histogramh2)
            distances.append(distance)
            j += 1

        find_similar_images(distances, train_images)


train_images = []
test_images = []
rgb_histograms = []
hsv_histograms = []

train_images, test_images = take_images(train_images, test_images)
rgb_histograms, hsv_histograms = save_histograms(train_images, rgb_histograms, hsv_histograms)

rgb_distance(train_images, test_images, rgb_histograms)
hue_distance(train_images, test_images, hsv_histograms)
