import logging

from PIL import Image
import os
import cv2
import numpy as np
import csv
import pandas as pd

logging.basicConfig(filename="project.log", level=logging.INFO)

def open_image(src):
    try:
        img = Image.open(src, 'r')
        logging.info('Image Open Success!')
    except Exception as e:
        logging.error("Image failed to open " + str(e))

    return img

def test(probability, path):
    im = open_image(path)
    temp = im.copy()
    image = create_image(temp, probability)
    return image


def create_image(im, probability):
    width, height = im.size
    try:
        pix = im.load()
        logging.info("load image successful!")
    except Exception as e:
        logging.error("Image loading failed!"+ str(e))

    try:
        for i in range(width):
            for j in range(height):
                r, g, b = im.getpixel((i, j))
                row_num = (r * 256 * 256) + (g * 256) + b  # calculating the serial row number
                if (probability['Probability'][row_num] < 0.555555):
                    pix[i, j] = (0, 0, 0)
                else:
                    pix[i, j] = (255, 255, 255)
    except Exception as e:
        logging.error(str(e))

    saveImage(im)
    return im


def saveImage(image):  ## saving image
    try:
        image.save('test/result.jpg')
        logging.info("Image saving successful!")
    except Exception as e:
        logging.error("Image saving failed!", str(e))

def main():
    print("Reading CSV...")
    probability = pd.read_csv('train.csv')  # getting the rows from csv
    print('Data collection completed')

    path = 'test/1.jpg'
    image = test(probability, path)  # this tests the data
    print("Image created")
    print(image)

if __name__ == "__main__":
    main()

#print(Image.open("test/1.jpg"))
#print(Image.open("test/result.jpg"))
