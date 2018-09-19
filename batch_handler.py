import gym
import random
import torch
import numpy as np
from wrapper import PreprocessingWrapper


class BatchHandler:
    def __init__(self, config):
        self.config = config
        self.environments = [PreprocessingWrapper(gym.make(name)) for name in config["atari_input_environments"]]
        self.batch_count = 0

    def _get_random_environment(self):
        return random.choice(self.environments)

    def get_next_batch(self, batch_size):
        batch = [env.reset() for env in self.environments]

        while True:
            env = self._get_random_environment()
            observation, reward, done, _ = env.step(env.action_space.sample())
            if np.mean(observation) > 0.01:
                batch.append(observation)
            if len(batch) == batch_size:
                self.batch_count += 1
                yield torch.tensor(np.array(batch, dtype=np.float32))
                batch.clear()
            if done:
                env.reset()
            if self.batch_count >= self.config["max_number_of_batches"]:
                self.reset()
                break

    def reset(self):
        for env in self.environments:
            env.close()

        self.environments = [PreprocessingWrapper(gym.make(name)) for name in self.config["atari_input_environments"]]
        self.batch_count = 0
