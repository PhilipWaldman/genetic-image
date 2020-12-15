import os

import cv2
from skimage import io


def save_video(frame_dirs):
    FPS = 24

    # TODO: needs to be saved without compression
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(os.path.join('.', 'output_data', 'video', 'genetic.avi'),
                            fourcc, float(FPS), io.imread(frame_dirs[0]).shape, isColor=False)

    for im_path in frame_dirs:
        im = io.imread(im_path)
        video.write(im)
    video.release()
