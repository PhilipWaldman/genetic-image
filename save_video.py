import os
from typing import Tuple

import cv2
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize


def save_video(frame_dirs, resolution: Tuple[int, int] = None):
    print('\nSaving as video...')
    FPS = 24

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    if resolution is None:
        video = cv2.VideoWriter(os.path.join('.', 'output_data', 'video', 'genetic.avi'),
                                fourcc, float(FPS), imread(frame_dirs[0]).shape, isColor=False)
    else:
        video = cv2.VideoWriter(os.path.join('.', 'output_data', 'video', 'genetic.avi'),
                                fourcc, float(FPS), resolution, isColor=False)

    for im_path in frame_dirs:
        if resolution is None:
            im = imread(im_path)
        else:
            im = img_as_ubyte(resize(imread(im_path, as_gray=True), resolution, anti_aliasing=False, order=0))
        video.write(im)
    video.release()
    print('Video saved.')
