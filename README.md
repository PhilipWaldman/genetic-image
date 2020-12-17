# Genetic Image
This program uses a genetic algorithm to generate an image that approaches the chosen training data. One or more training images can be chosen to train to. An initial image can also be chosen. The initial image is the image every `Image` in the population has at generation 0. If no initial image is specified, it will be initialized with random noise. If a generation has an `Image` with a higher fitness, it will be saved as a `.png`. At the end all these saved images will be combined into a video.

## Genetic Algorithm
The genetic algorithm I use here is slightly different from the standard genetic algorithm. I did this to improve the terrible performance slightly and to add the feature of saving the current best image. This is how the modified genetic algorithm works in this program:

First, a `Population` is created that stores a list of n `Image` objects. These images are either initialize with noise or an initial image. Then, it is evolution time! The way natural selection works is that first the fitness of every `Image` is calculated. A higher fitness means that it is a better approximation of the target image. All these fitnesses are summed together for later use. The first `Image` to be added to the next generation is the `Image` that had the highest fitness and, thus, is the best. The next `Image` to be added to the next generation is a randomly selected `Image` from the current generation weighted by fitness, meaning that those with a higher fitness are more likely to be chosen. Then, it is mutated. When mutating, every pixel has a `mutation_rate` chance of being assigned a random value. These `Image` selection and mutation steps are repeated n-1 times to make the next generation's population the same size as the current generation's. Lastly, the current generation is replaced by the next generation.

## Training Data
This is the data that the user provides for the algorithm to use. All the training data images have to be the same size. This is ensured by using the `resize()` function.

### Target Image(s)
These are the image the algorithm should approach. They should be stored in a separate folder because all images in the specified folder will be used.

### Initial Image
This is the images every `Image` in the population will start with. This can be done to slowly transform the initial image into the target image. It is not required to specify the initial image. If it isn't specified, random noise will be used.

## Output Data
The program save images and a video to the `output_data` directory. These files can be manually analysed.

### Images
 After natural selection has been preformed, the current best `Image` is compared with the previous. If the current `Image` is different, it will be saved to `gen #.png`, where # is the current generation. This is so the best `Image` can be inspected for each generation.

### Video
After the genetic algorithm has finished, all the images that were saved get compiled into a `.avi` video. This make it easier to see a transition from the initial image, either noise or a predefined image, to the target image.

I could not figure out how to save the video without compression. This makes it harder to see what the frames are showing if the resolution is very low.

## Running Time Analysis
For tiny images, 8x8 pixels, it is slow (order of minutes). For small images, 32x32 pixels, it is really, really slow (order of hours). For anything larger... don't even try.

## Installation
First the required packages have to be installed. This can be done by running `pip install -r requirements.txt`. Then, to actually run the tool, run `python3 main.py` or run `main.py` in your favorite IDE.

### Packages used
* numpy
* scikit-image
* opencv-python

## Usage

