import numpy as np
import cv2
import math
import random


class Coloring:

    def intensity_slicing(self, image, n_slices):
        # Convert greyscale image to color image using color slicing technique.
        # takes as input:
        # image: the grayscale input image
        # n_slices: number of slices

        # Steps:

        # 1. Split the exising dynamic range (0, k-1) using n slices (creates n+1 intervals)
        # 2. Randomly assign a color to each interval
        # 3. Create and output color image
        # 4. Iterate through the image and assign colors to the color image based on which interval the intensity belongs to

        # returns color image

        size = n_slices + 1
        interval = 256 / size
        shape = np.shape(image)
        color = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

        red = [0] * size
        green = [0] * size
        blue = [0] * size

        for i in range(1, size):
            red[i] = random.randint(0, 255)
            green[i] = random.randint(0, 255)
            blue[i] = random.randint(0, 255)

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                intensity = image[i][j]
                index = int(intensity / interval)
                color[i][j] = [red[index], green[index], blue[index]]

        return color

    def color_transformation(self, image, n_slices, theta):
        # Convert greyscale image to color image using color transformation technique.
        # takes as input:
        # image:  grayscale input image
        # colors: color array containing RGB values

        # Steps:
        # 1. Split the exising dynamic range (0, k-1) using n slices (creates n+1 intervals)
        # 2. create red values for each slice using 255*sin(slice + theta[0])
        #    similarly create green and blue using 255*sin(slice + theta[1]), 255*sin(slice + theta[2])
        # 3. Create and output color image
        # 4. Iterate through the image and assign colors to the color image based on which interval the intensity belongs to

        # returns color image

        size = n_slices + 1
        interval = 256 / size
        shape = np.shape(image)
        color = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

        red = [0] * size
        green = [0] * size
        blue = [0] * size

        for i in range(1, size):
            red[i] = 255*np.sin(i+(theta[0]*math.pi/180))
            green[i] = 255*np.sin(i+(theta[1]*math.pi/180))
            blue[i] = 255*np.sin(i+(theta[2]*math.pi/180))

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                intensity = image[i][j]
                index = int(intensity / interval)
                color[i][j] = [red[index], green[index], blue[index]]

        return color

        return image
