from pathlib import Path
import tensorflow as tf
import numpy as np
import train
from PIL import Image, ImageFilter
import cv2
import random

CHECKPOINT_PATH = f'./checkpoints/'
SAVE_PATH = f'./new_samples/'
NUM_TO_GENERATE = 100


def _create_necessary_directories(paths: [Path]) -> None:
    for directory in paths:
        if not Path(directory).exists():
            Path(directory).mkdir()


def get_samples(generator: train.Generator, n: int = 1, size: (int, int) = None, mean: float = 0.0, stddev: float = 1.0) -> [Image]:
    noise = tf.random.normal([n, 100], mean=mean, stddev=1.0)
    predictions = generator.model(noise, training=False)
    image_objects = []
    for i in range(n):
        image_tensor = predictions[i, :, :, :] * 127.5 + 127.5
        image_array = np.array(image_tensor).astype('uint8')
        image_array = cv2.fastNlMeansDenoisingColored(image_array)
        image = Image.fromarray(image_array)
        if size:
            image = image.resize(size)
        image_objects.append(image)
    return image_objects


if __name__ == '__main__':
    _create_necessary_directories([CHECKPOINT_PATH, SAVE_PATH])
    print('Creating models, losses, and optimizers')
    generator = train.Generator()
    discriminator = train.Discriminator()

    print("Initializing and restoring from checkpoint")
    checkpoint_prefix = str(Path(CHECKPOINT_PATH) / 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer,
                                     discriminator_optimizer=discriminator.optimizer,
                                     generator=generator.model,
                                     discriminator=discriminator.model)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))

    samples = get_samples(generator, n=NUM_TO_GENERATE, size=(
        256, 256), mean=0.0, stddev=1.0)
    for i, sample in enumerate(samples):
        sample.save(SAVE_PATH+f'sample_{i:04}.png')
