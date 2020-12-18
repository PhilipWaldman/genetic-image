import os
from typing import Tuple

import cv2
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize


def save_video(frame_dirs: list, resolution: Tuple[int, int] = None):
    """ Save the frames that are stored at the locations in the specified list as a video at 24 fps with either the
    resolution of the frames or the specified resolution. The video is saved as a .avi with MP42 encoding (I think, I
    don't understand the encoding stuff).

    :param frame_dirs: The list of directories of the images to save as a video.
    :param resolution: The resolution to save the video as. Default: the resolution of the images at the locations in
    frame_dirs.
    """
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
