import abc

import numpy as np


class Policy(abc.ABC):
    """
    Policy base class without Parameterzied.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """
    def __init__(self, env_spec):
        self._env_spec = env_spec

    @abc.abstractmethod
    def get_action(self, observation):
        """Get action given observation."""
        pass

    def get_actions(self, observations):
        """Get actions given observations."""
        return np.stack([self.get_action(obs) for obs in observations])

    @property
    def observation_space(self):
        """Observation space."""
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """Policy action space."""
        return self._env_spec.action_space

    @property
    def env_spec(self):
        """Policy environment specification."""
        return self._env_spec

    @property
    def recurrent(self):
        """Boolean indicating if the policy is recurrent."""
        return False
