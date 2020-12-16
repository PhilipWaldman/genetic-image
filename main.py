import os
import random
import time

import numpy as np
from skimage import io
from skimage.transform import resize

from save_video import save_video


class Image:
    def __init__(self, shape: tuple, initial_image=None):
        """ Initializes an image of the specified shape. If an initial image is enter, that will be set as the image.
        It initial_image is left empty, an image with random noise will be generated.

        :param shape: A tuple of the shape of the image. (# of rows, # of cols)
        :param initial_image: A numpy array of the image to start as.
        """
        if initial_image:
            self.image = initial_image
        else:
            self.image = np.random.random(size=shape)
        self.fitness = 0

    def calc_fitness(self):
        """ Calculates the fitness of the image and assigns it to self.fitness. """
        self.fitness = 0
        for im in training_images:
            for r in range(len(im)):
                for c in range(len(im[r])):
                    self.fitness += 1 - abs(im[r][c] - self.image[r][c])

    def mutate(self):
        """ Mutates the image. Every pixel in the image has a mutation_rate chance of being set to a random value. """
        for r in range(len(self.image)):
            for c in range(len(self.image[r])):
                if random.random() < mutation_rate:
                    self.image[r][c] = random.random()

    def save_image(self, name: str) -> str:
        """ Saves the image with the specified name and returns where it is saved.

        :param name: The name of the image.
        :return: Where the image is saved. With its directory and file name.
        """
        save_to = os.path.join(image_save_path, f'{name}.png')
        io.imsave(save_to, (self.image * 255).astype(np.uint8))
        return save_to


class Population:
    def __init__(self, pop_size: int, shape):
        """ Initializes a population of specified size of images with specified size.

        :param pop_size: The population size.
        :param shape: The shape of the images in the population.
        """
        self.images = [Image(shape) for _ in range(pop_size)]
        self.fitness_sum = 0

    def natural_selection(self):
        """ Preforms natural selection in the population. """
        self.calculate_fitnesses()
        new_images = [self.best_image()]
        for i in range(len(self.images) - 1):
            new_images.append(self.select_parent())
            new_images[-1].mutate()
        self.images = new_images.copy()

    def best_image(self):
        """ Finds and returns the image with the highest fitness.

        :return: The best image in the population.
        """
        best_im = Image((1, 1))
        for image in self.images:
            if image.fitness > best_im.fitness:
                best_im = image
        return best_im

    def calculate_fitnesses(self):
        """ Calculates the fitnesses of all the images and computes the sum of all the fitnesses. """
        self.fitness_sum = 0
        for image in self.images:
            image.calc_fitness()
            self.fitness_sum += image.fitness

    def select_parent(self):
        """ Selects a random parent weighted by their fitness.

        :return: A randomly selected image from the population.
        """
        rand = random.random() * self.fitness_sum
        running_sum = 0
        for image in self.images:
            running_sum += image.fitness
            if running_sum >= rand:
                return image


def calc_accuracy():
    return pop.images[0].fitness / (pop.images[0].image.shape[0] * pop.images[0].image.shape[1] * len(training_images))


def train_generation(gen: int, prev_best: Image) -> tuple:
    """

    :param gen: The current generation number.
    :param prev_best: The best image of the previous generation.
    :return: A tuple of the best image of the current generation and its accuracy.
    """
    print(f'Training generation {gen}...')
    pop.natural_selection()
    acc = calc_accuracy()
    if False in (prev_best.image == pop.images[0].image):
        saved_to = pop.images[0].save_image(f'gen {gen}')
        frame_dirs.append(saved_to)
        print(f'Best fitness: {pop.images[0].fitness}')
        print(f'Accuracy: {acc}')
    return pop.images[0], acc


def train_generations(generations: int):
    """ Trains for the specified number of generations.

    :param generations: The number of generations to train for.
    """
    print(f'Going to train for {generations} generations...')
    prev_best = pop.images[0]
    for gen in range(1, generations + 1):
        prev_best, better_acc = train_generation(gen, prev_best)


def train_accuracy(accuracy: float):
    """ Trains until the specified accuracy is reached.\n
    The accuracy is calculated by dividing the fitness by the maximum fitness possible. Thus, an accuracy of 1 means
    that the generated image is exactly the same at the target image.

    :param accuracy: The accuracy to train to. Float in range [0, 1].
    """
    print(f'Going to train until the accuracy has reached {accuracy}...')
    prev_best = pop.images[0]
    gen = 0
    acc = calc_accuracy()
    while acc < accuracy:
        gen += 1
        better_image, better_acc = train_generation(gen, prev_best)
        if better_image and better_acc:
            prev_best = better_image
            acc = better_acc


def train_time(seconds: int, minutes=0, hours=0):
    """ Trains for the time specified.\n
    The time to train is the sum of all parameters. Specifically, seconds + minutes * 60 + hours * 3600

    :param seconds: The amount of seconds to train.
    :param minutes: The amount of minutes to train. Default = 0
    :param hours: The amount of hours to train. Default = 0
    """
    time_to_train = seconds + minutes * 60 + hours * 60 * 60
    print(f'Going to train for {seconds} seconds, {minutes} minutes, and {hours} hours...')
    t_start = time.time()
    prev_best = pop.images[0]
    gen = 0
    while time.time() - t_start < time_to_train:
        gen += 1
        prev_best, better_acc = train_generation(gen, prev_best)


image_save_path = os.path.join('output_data', 'images')
frame_dirs = []
training_folder = os.path.join('training_data', 'triangle')
training_paths = [os.path.join(training_folder, item) for item in os.listdir(training_folder)]
training_images = [resize(io.imread(path, as_gray=True), (32, 32)) for path in training_paths]

pop = Population(100, training_images[0].shape)
mutation_rate = 1 / 100

t0 = time.perf_counter()
# train_generations(100)
# train_accuracy(0.75)
train_time(10)
t1 = time.perf_counter()

t = t1 - t0
hrs = t // (60 * 60)
mins = (t - hrs * 60 * 60) // 60
secs = (t - hrs * 60 * 60) - mins * 60
print(f'\nTrained for {round(t, 3)} seconds.\n'
      f'That is {hrs} hours, {mins} minutes, and {round(secs, 3)} seconds.')

save_video(frame_dirs)
