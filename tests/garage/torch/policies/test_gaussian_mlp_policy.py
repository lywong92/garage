import pickle
import unittest
from unittest import mock

import nose2.tools.params as params
import numpy as np
import torch

from garage.torch.policies import GaussianMLPPolicy


class TestGaussianMLPPolicy(unittest.TestCase):

    @mock.patch('garage.torch.policies.gaussian_mlp_policy.GaussianMLPModule')
    def test_policy_get_actions(self, mock_model):
        action, mean, log_std = (torch.Tensor(x) for x in ([5.0, 3., 2.], [0.5, 0.2, 0.4], [0.25, 0.11, 0.44]))
        mock_model.return_value = lambda x: (action, mean, log_std, None, None)
        input_dim, output_dim, hidden_sizes = (5, 3, (2, 2))

        env_spec = mock.MagicMock()
        env_spec.observation_space.flat_dim = input_dim
        env_spec.action_space.flat_dim = output_dim

        policy = GaussianMLPPolicy(env_spec, hidden_sizes=hidden_sizes)

        input = torch.ones(input_dim)
        sample, dist_info = policy.get_actions(input)

        assert np.array_equal(sample, action)
        assert np.array_equal(dist_info['mean'], mean)
        assert np.array_equal(dist_info['log_std'], log_std)

    @mock.patch('garage.torch.policies.gaussian_mlp_policy.GaussianMLPModule')
    def test_policy_get_action(self, mock_model):
        action, mean, log_std = (torch.Tensor(x) for x in ([5.0, 3., 2.], [0.5, 0.2, 0.4], [0.25, 0.11, 0.44]))
        mock_model.return_value = lambda x: (action, mean, log_std, None, None)
        input_dim, output_dim, hidden_sizes = (5, 3, (2, 2))

        env_spec = mock.MagicMock()
        env_spec.observation_space.flat_dim = input_dim
        env_spec.action_space.flat_dim = output_dim

        policy = GaussianMLPPolicy(env_spec, hidden_sizes=hidden_sizes)

        input = torch.ones(input_dim)
        sample, dist_info = policy.get_action(input)

        assert sample == action[0].item()
        assert dist_info['mean'] == mean[0].item()
        assert dist_info['log_std'] == log_std[0].item()

    @mock.patch('torch.rand')
    @params(
        (5, 1, (1,)),
        (5, 1, (2,)),
        (5, 2, (3,)),
        (5, 2, (1, 1)),
        (5, 3, (2, 2)))
    def test_policy_is_picklable(self, input_dim, output_dim, hidden_sizes, mock_normal):

        mock_normal.return_value = 0.5

        env = TestEnv(TestSpace(input_dim), TestSpace(output_dim))

        policy = GaussianMLPPolicy(env, hidden_sizes=hidden_sizes)

        input = torch.ones(input_dim)
        sample, dist_info = policy.get_actions(input)

        h = pickle.dumps(policy)
        policy_pickled = pickle.loads(h)

        sample_p, dist_info_p = policy_pickled.get_actions(input)

        assert np.array_equal(sample, sample_p)
        assert np.array_equal(dist_info['mean'], dist_info_p['mean'])
        assert np.array_equal(dist_info['log_std'], dist_info_p['log_std'])


class TestEnv():
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class TestSpace():
    def __init__(self, val):
        self.flat_dim = val
