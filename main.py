import os
import random
import time

import numpy as np
from skimage import io
from skimage.transform import resize

from save_video import save_video


class Image:
    def __init__(self, shape):
        self.image = np.random.random(size=shape)
        self.fitness = 0

    def calc_fitness(self):
        self.fitness = 0
        for im in training_images:
            for r in range(len(im)):
                for c in range(len(im[r])):
                    self.fitness += 1 - abs(im[r][c] - self.image[r][c])

    def mutate(self, mutation_rate):
        for r in range(len(self.image)):
            for c in range(len(self.image[r])):
                if random.random() < mutation_rate:
                    self.image[r][c] = random.random()

    def save_image(self, name):
        io.imsave(name, (self.image * 255).astype(np.uint8))

    def copy(self):
        im = Image(self.image.shape)
        im.image = self.image.copy()
        im.fitness = self.fitness
        return im


class Population:
    def __init__(self, pop_size, shape):
        self.images = [Image(shape) for _ in range(pop_size)]
        self.fitness_sum = 0

    def natural_selection(self, mutation_rate):
        self.calculate_fitnesses()
        new_images = [self.best_image()]
        for i in range(len(self.images) - 1):
            new_images.append(self.select_parent())
            new_images[-1].mutate(mutation_rate)
        self.images = new_images.copy()

    def best_image(self):
        best_im = Image((1, 1))
        for image in self.images:
            if image.fitness > best_im.fitness:
                best_im = image
        return best_im.copy()

    def calculate_fitnesses(self):
        self.fitness_sum = 0
        for image in self.images:
            image.calc_fitness()
            self.fitness_sum += image.fitness

    def select_parent(self):
        rand = random.random() * self.fitness_sum
        running_sum = 0
        for image in self.images:
            running_sum += image.fitness
            if running_sum >= rand:
                return image.copy()


def train_gens(generations, mutation_rate):
    prev_best = pop.images[0]
    for gen in range(generations):
        print(f'Training generation {gen}...')
        pop.natural_selection(mutation_rate)
        if False in (prev_best.image == pop.images[0].image):
            prev_best = pop.images[0]
            pop.images[0].save_image(os.path.join(image_save_path, f'gen {gen}.png'))
            print(f'Best fitness: {pop.images[0].fitness}')


def train_acc(accuracy, mutation_rate):
    frame_dirs = []
    prev_best = pop.images[0]
    gen = 0
    acc = pop.images[0].fitness / (pop.images[0].image.shape[0] * pop.images[0].image.shape[1])
    while acc < accuracy:
        gen += 1
        print(f'Training generation {gen}...')
        pop.natural_selection(mutation_rate)
        if False in (prev_best.image == pop.images[0].image):
            acc = (pop.images[0].fitness / (pop.images[0].image.shape[0] * pop.images[0].image.shape[1]))
            prev_best = pop.images[0]
            pop.images[0].save_image(os.path.join(image_save_path, f'gen {gen}.png'))
            frame_dirs.append(os.path.join(image_save_path, f'gen {gen}.png'))
            print(f'Best fitness: {pop.images[0].fitness}')
            print(f'Accuracy: {acc}')
    return frame_dirs


image_save_path = os.path.join('output_data', 'images')
training_folder = os.path.join('training_data', 'grid')
training_paths = [os.path.join(training_folder, item) for item in os.listdir(training_folder)]
training_images = [resize(io.imread(path, as_gray=True), (8,8)) for path in training_paths]
pop = Population(100, training_images[0].shape)

t0 = time.perf_counter()
# train_gens(10000, 1 / 1024)
frame_locs = train_acc(0.7, 1 / 64)
t1 = time.perf_counter()
t = t1 - t0
print(f'\nTrained for {round(t, 3)} seconds\n'
      f'That is {round(t / 60, 3)} minutes\n'
      f'Or {round(t / (60 * 60), 3)} hours')

save_video(frame_locs)
