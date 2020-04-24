from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Image Constants
CHANNELS = 3
SIZE = (64, 64)

# Data Constants
BATCH_SIZE = 64
BUFFER_SIZE = 10000

# Training Constants
EPOCHS = 500
LEARNING_RATE = 0.0002
LABEL_SMOOTH = 0.9

# Generation Constants
NOISE_DIM = 100
NUM_OF_EXAMPLES = 16
SAMPLE_EVERY = 1
SAVE_EVERY = 10
START_SAVING = 90
RANDOM_VECTOR_FOR_GENERATION = tf.random.normal(
    [NUM_OF_EXAMPLES, NOISE_DIM]
)

# Model Constants
DROPOUT_D_RATE = 0.5
BATCH_NORM_MOMENTUM = 0.8
KERNEL_SIZE = (4, 4)
GENERATOR_STRIDE_SIZE = (1, 1)
DISCRIMINATOR_STRIDE_SIZE = (2, 2)
PADDING = "same"  # valid or same
USE_BIAS = False

# Directories
DATA_PATH = "./data"
SAMPLES_PATH = f'./samples/'
CHECKPOINT_PATH = f'./checkpoints/'


def _create_necessary_directories(paths: [Path]) -> None:
    for directory in paths:
        if not Path(directory).exists():
            Path(directory).mkdir()


def _get_dataset(image_path: Path) -> tf.data.Dataset:
    image_tensors = []
    for image_path in Path(image_path).iterdir():
        image_raw = tf.io.read_file(str(image_path))
        image_tensor = tf.io.decode_image(
            image_raw, channels=CHANNELS, dtype=tf.float32)
        image_tensor = tf.image.resize(image_tensor, SIZE)
        image_tensor = image_tensor * 2.0 - 1.0
        image_tensors.append(image_tensor)

    dataset = tf.data.Dataset.from_tensor_slices(
        image_tensors).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    return dataset


class Generator:
    def __init__(self):
        self.model = self._create_model()
        self.loss = self._create_loss()
        self.optimizer = self._create_optimizer()

    def _create_model(self) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            4*4*512, use_bias=False, input_shape=(NOISE_DIM,)))
        self._add_batch_norm(model)
        self._add_activation(model)
        model.add(tf.keras.layers.Reshape((4, 4, 512)))

        self._add_upsampling(model)
        self._add_conv_2d_transpose(model, 512, (1, 1))
        self._add_batch_norm(model)
        self._add_activation(model)

        self._add_upsampling(model)
        self._add_conv_2d_transpose(model, 256)
        self._add_batch_norm(model)
        self._add_activation(model)

        self._add_upsampling(model)
        self._add_conv_2d_transpose(model, 128)
        self._add_batch_norm(model)
        self._add_activation(model)

        self._add_upsampling(model)
        self._add_conv_2d_transpose(model, 64)
        self._add_batch_norm(model)
        self._add_activation(model)

        self._add_conv_2d_transpose(model, CHANNELS)
        self._add_last_activation(model)

        return model

    def _add_conv_2d_transpose(self, model: tf.keras.Sequential, filters: int, strides: (int, int) = GENERATOR_STRIDE_SIZE) -> None:
        model.add(tf.keras.layers.Conv2DTranspose(filters, KERNEL_SIZE, strides=strides, padding=PADDING,
                                                  use_bias=USE_BIAS, kernel_initializer=tf.random_normal_initializer(0.0, 0.02)))

    def _add_batch_norm(self, model: tf.keras.Sequential) -> None:
        model.add(tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_MOMENTUM,
            gamma_initializer=tf.random_normal_initializer(1.0, 0.02)))

    def _add_upsampling(self, model: tf.keras.Sequential) -> None:
        model.add(tf.keras.layers.UpSampling2D())

    def _add_activation(self, model: tf.keras.Sequential) -> None:
        model.add(tf.keras.layers.ReLU())

    def _add_last_activation(self, model: tf.keras.Sequential) -> None:
        model.add(tf.keras.layers.Activation('tanh'))

    def _create_loss(self) -> tf.losses:
        return tf.losses.binary_crossentropy

    def _create_optimizer(self) -> tf.keras.optimizers:
        return tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.99)


class Discriminator:
    def __init__(self):
        self.model = self._create_model()
        self.loss = self._create_loss()
        self.optimizer = self._create_optimizer()

    def _create_model(self) -> tf.keras.Sequential:
        model = tf.keras.Sequential()

        self._add_conv_2d(model, 64)
        self._add_activation(model)
        self._add_dropout(model)

        self._add_conv_2d(model, 128)
        self._add_batch_norm(model)
        self._add_activation(model)
        self._add_dropout(model)

        self._add_conv_2d(model, 256)
        self._add_batch_norm(model)
        self._add_activation(model)
        self._add_dropout(model)

        self._add_conv_2d(model, 512)
        self._add_batch_norm(model)
        self._add_activation(model)
        self._add_dropout(model)

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation('sigmoid'))

        return model

    def _add_conv_2d(self, model: tf.keras.Sequential, filters: int, strides: (int, int) = DISCRIMINATOR_STRIDE_SIZE) -> None:
        model.add(tf.keras.layers.Conv2D(filters, KERNEL_SIZE, strides=strides, padding=PADDING,
                                         use_bias=USE_BIAS, kernel_initializer=tf.random_normal_initializer(0.0, 0.02)))

    def _add_batch_norm(self, model: tf.keras.Sequential) -> None:
        model.add(tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_MOMENTUM,
            gamma_initializer=tf.random_normal_initializer(1.0, 0.02)))

    def _add_dropout(self, model: tf.keras.Sequential) -> None:
        model.add(tf.keras.layers.Dropout(rate=DROPOUT_D_RATE))

    def _add_activation(self, model: tf.keras.Sequential) -> None:
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    def _create_loss(self) -> tf.losses:
        return tf.losses.binary_crossentropy

    def _create_optimizer(self) -> tf.keras.optimizers:
        return tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.99)


def _train(generator: Generator, discriminator: Discriminator, dataset: tf.data.Dataset, epochs: int, checkpoint: tf.train.Checkpoint, checkpoint_prefix: str) -> None:
    _generate_and_save_images(generator, 0, RANDOM_VECTOR_FOR_GENERATION)
    for epoch in range(epochs):
        start = time.time()
        total_discriminator_loss, total_generator_loss = _train_epoch(
            generator, discriminator, dataset)

        print(
            f"Epoch# {epoch} TotalEpochs {epochs} TimeTaken: {time.time()-start} DLoss {total_discriminator_loss} GLoss {total_generator_loss}")

        if epoch % SAMPLE_EVERY == 0:
            print(f"Generating and saving image for epoch {epoch}")
            _generate_and_save_images(
                generator, epoch + 1, RANDOM_VECTOR_FOR_GENERATION)
            print("Image saved")
        if epoch >= START_SAVING and epoch % SAVE_EVERY == 1:
            print(f"Saving checkpoint for epoch {epoch}")
            _save_checkpoint(checkpoint, checkpoint_prefix)
            print('checkpoint saved')


def _train_epoch(generator: Generator, discriminator: Discriminator, dataset: tf.data.Dataset) -> (float, float):
    d_loss, g_loss = 0.0, 0.0
    for image_batch in dataset:
        d_batch_loss = _train_discriminator(discriminator, image_batch)
        g_batch_loss = _train_generator(generator, discriminator)
        d_loss += d_batch_loss
        g_loss += g_batch_loss
    return d_loss, g_loss


def _train_discriminator(discriminator: Discriminator, image_batch: [tf.image]) -> float:
    with tf.GradientTape() as real_tape:
        image_batch = image_batch + tf.random.normal(shape=tf.shape(image_batch), mean=0.0,
                                                     stddev=0.1, dtype=tf.float32)
        real_outputs = discriminator.model(image_batch, training=True)
        real_loss = discriminator.loss(tf.ones_like(
            real_outputs)*LABEL_SMOOTH, real_outputs)
    _calculate_and_apply_gradients(
        discriminator.model, discriminator.optimizer, real_tape, real_loss)

    with tf.GradientTape() as fake_tape:
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        fake_images = generator.model(noise, training=True)
        fake_outputs = discriminator.model(fake_images, training=True)
        fake_loss = discriminator.loss(tf.zeros_like(
            fake_outputs), fake_outputs)
    _calculate_and_apply_gradients(
        discriminator.model, discriminator.optimizer, fake_tape, fake_loss)
    d_loss = sum(real_loss)/len(real_loss) + sum(fake_loss)/len(fake_loss)

    return d_loss


def _train_generator(generator: Generator, discriminator: Discriminator) -> float:
    with tf.GradientTape() as gen_tape:
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        fake_images = generator.model(noise, training=True)
        gen_outputs = discriminator.model(fake_images, training=True)

        gen_loss = generator.loss(
            tf.ones_like(gen_outputs), gen_outputs)

    g_loss = sum(gen_loss)/len(gen_loss)
    _calculate_and_apply_gradients(
        generator.model, generator.optimizer, gen_tape, gen_loss)
    return g_loss


def _calculate_and_apply_gradients(model: tf.keras.Sequential, optimizer: tf.keras.optimizers, gradient_tape: tf.GradientTape, loss: [float]):
    gradients = gradient_tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def _generate_and_save_images(generator: Generator, epoch: int, sample_input: [int, int]) -> None:
    predictions = generator.model(sample_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(SAMPLES_PATH+'/epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close(fig)


def _save_checkpoint(checkpoint: tf.train.Checkpoint, checkpoint_prefix) -> None:
    checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":

    print('Creating models, losses, and optimizers')
    generator = Generator()
    discriminator = Discriminator()

    print('Creating necessary directories')
    _create_necessary_directories([DATA_PATH, SAMPLES_PATH, CHECKPOINT_PATH])

    print('Processing data and generating dataset')
    dataset = _get_dataset(DATA_PATH)

    print("Initializing and restoring from checkpoint")
    checkpoint_prefix = str(Path(CHECKPOINT_PATH) / 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer,
                                     discriminator_optimizer=discriminator.optimizer,
                                     generator=generator.model,
                                     discriminator=discriminator.model)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))

    print("Starting training")
    _train(generator, discriminator, dataset,
           EPOCHS, checkpoint, checkpoint_prefix)
