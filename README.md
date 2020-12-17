# Genetic Image
This program uses a genetic algorithm to generate an image that approaches the chosen training data. One or more training images can be chosen to train to. An initial image can also be chosen. The initial image is the image every `Image` in the population has at generation 0. If no initial image is specified, it will be initialized with random noise. If a generation has an `Image` with a higher fitness, it will be saved as a `.png`. At the end all these saved images will be combined into a video.

## Genetic Algorithm


## Training Data
This is the data that the user provides for the algorithm to use. All the training data images have to be the same size. This is ensured by using the `resize()` function.

### Target Image(s)
These are the image the algorithm should approach. They should be stored in a separate folder because all images in the specified folder will be used.

### Initial Image
This is the images every `Image` in the population will start with. This can be done to slowly transform the initial image into the target image. It is not required to specify the initial image. If it isn't specified, random noise will be used.

## Output Data


### Images


### Video