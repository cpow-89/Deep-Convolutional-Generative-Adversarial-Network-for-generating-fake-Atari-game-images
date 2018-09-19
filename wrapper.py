import gym
import cv2
import numpy as np


IMAGE_SIZE = 64


class PreprocessingWrapper(gym.ObservationWrapper):
    """Preprocessing of input numpy array
    Step 1: resize image into predefined size
    Step 2: move color channel axis to athe first place to satisfy pytorch conventions
    Step 3: normalize"""
    def __init__(self, *args):
        super().__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(old_space.low),
                                                self.observation(old_space.high),
                                                dtype=np.float32)

    def observation(self, observation):
        new_obs = self._resize_image(observation)
        new_obs = self._transform(new_obs)
        new_obs = self._normalize(new_obs)
        return new_obs

    def _resize_image(self, observation):
        return cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))

    def _transform(self, observation):
        return np.moveaxis(observation, 2, 0)

    def _normalize(self, observation):
        return observation.astype(np.float32) / 255.0

