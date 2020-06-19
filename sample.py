import os
import dcgan

CHECKPOINT_PATH = './checkpoints/'
SAMPLES_PATH = './images/'
NUM_TO_GENERATE = 100

if __name__ == '__main__':
    sampler = dcgan.Sampler(CHECKPOINT_PATH)
    if not os.path.exists(SAMPLES_PATH):
        os.mkdir(SAMPLES_PATH)
    for i in range(NUM_TO_GENERATE):
        sample = sampler.sample()
        sample.save(os.path.join(SAMPLES_PATH, f'{i}.png'))
