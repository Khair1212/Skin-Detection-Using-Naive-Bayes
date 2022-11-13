from PIL import Image
import os
import cv2
import numpy as np
import csv
import pandas as pd

import logging

logging.basicConfig(filename="project.log", level=logging.INFO)


def open_image(src):
    try:
        img = Image.open(src, 'r')
        logging.info('Image Open Success!')
    except Exception as e:
        logging.error("Image failed to open " + str(e))

    return img


def image_to_pixel(image_path, mask_path):
    # image = image.convert('RGB')
    # return list(image.getdata())
    # image = image.convert('RGB')
    # return list(image.getdata())
    try:
        pixel_image = list(open_image(image_path).convert('RGB').getdata())
        pixel_mask = list(open_image(mask_path).convert('RGB').getdata())
        logging.info("Image Pixel Conversion Success!")
        #print(image_path, mask_path)
        return pixel_image, pixel_mask
    except Exception as e:
        logging.error("Pixel conversion failed " + str(e))


def is_skin(r, g, b):
    if (r <= 150 and g <= 150 and b <= 150):
        return False
    else:
        return True


def training(pixels, pixel_image, pixel_mask, skin, non_skin):
    try:
        for i in range(len(pixel_image)):
            r = pixel_image[i][0]
            g = pixel_image[i][1]
            b = pixel_image[i][2]

            if is_skin(r, g, b):
                skin[r][g][b] += 1
            else:
                non_skin[r][g][b] += 1
        logging.info("Training Successful")
        return pixels, skin, non_skin
    except Exception as e:
        logging.error("Training failed " + str(e))



def to_list(r, g, b, probability):
    a = []
    a.append(r)
    a.append(g)
    a.append(b)
    a.append(probability)
    return list(a)


def set_probability(pixel, skin, non_skin, probability):
    probability = list(skin / (non_skin + skin))
    return probability


def data(probability):
    arr = []
    for r in range(256):
        for g in range(256):
            for b in range(256):
                arr.append(to_list(r, g, b, probability[r][g][b]))
    return arr


def create_csv(probability):
    try:
        myFile = open('train.csv', 'w', newline='')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(["Red", "Green", "Blue", "Probability"])
            writer.writerows(data(probability))
        logging.info("CSV Creation Success!")
    except Exception as e:
        logging.error("CSV Creation failed " + str(e))

    print('Training Completed')


def main():
    pixels = np.zeros([256, 256, 256])
    skin = np.zeros([256, 256, 256])
    non_skin = np.zeros([256, 256, 256])
    probability = np.zeros((256, 256, 256))

    image_directory = sorted(os.listdir('image/'))
    mask_directory = sorted(os.listdir("mask/"))
    for i in range(1, len(image_directory)):
        # print(image_directory[i])
        image_path = "image/" + image_directory[i]
        mask_path = "mask/" + mask_directory[i]

        # reading the images and get the pixels
        image_pixels, mask_pixels = image_to_pixel(image_path, mask_path)

        # training
        pixels, skin, non_skin = training(pixels, image_pixels, mask_pixels, skin, non_skin)

    # Get the probabilities
    probability = set_probability(pixels, skin, non_skin, probability)
    create_csv(probability)


if __name__ == "__main__":
    main()
