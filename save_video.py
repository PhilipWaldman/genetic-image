import os

import cv2
import numpy as np
from skimage import io


def save_video(frames):
    width = 32
    height = 32
    FPS = 24

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('./genetic.avi', fourcc, float(FPS), (width, height))

    output_folder = 'C:/Users/Philip/Desktop/Python/Genetic Image/output_data'
    for item in os.listdir(output_folder):
        im_path = os.path.join(output_folder, item)
        im = io.imread(im_path)
        frame = np.array([[[c] * 3 for c in r] for r in im])
        video.write(frame)
    video.release()
