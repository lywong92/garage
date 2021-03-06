import pickle
from unittest import mock

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalGRUPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.models import SimpleGRUModel


class TestCategoricalGRUPolicyWithModel(TfGraphTestCase):
    @params(
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    )
    def test_dist_info_sym(self, obs_dim, action_dim, hidden_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))

        obs_ph = tf.placeholder(
            tf.float32, shape=(None, None, env.observation_space.flat_dim))

        with mock.patch(('garage.tf.policies.'
                         'categorical_gru_policy_with_model.GRUModel'),
                        new=SimpleGRUModel):
            policy = CategoricalGRUPolicyWithModel(
                env_spec=env.spec, state_include_action=False)

        policy.reset()
        obs = env.reset()

        dist_sym = policy.dist_info_sym(
            obs_var=obs_ph, state_info_vars=None, name='p2_sym')
        dist = self.sess.run(
            dist_sym, feed_dict={obs_ph: [[obs.flatten()], [obs.flatten()]]})

        assert np.array_equal(dist['prob'], np.full((2, 1, action_dim), 0.5))

    @params(
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    )
    def test_dist_info_sym_include_action(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))

        obs_ph = tf.placeholder(
            tf.float32, shape=(None, None, env.observation_space.flat_dim))

        with mock.patch(('garage.tf.policies.'
                         'categorical_gru_policy_with_model.GRUModel'),
                        new=SimpleGRUModel):
            policy = CategoricalGRUPolicyWithModel(
                env_spec=env.spec, state_include_action=True)

        policy.reset()
        obs = env.reset()

        dist_sym = policy.dist_info_sym(
            obs_var=obs_ph,
            state_info_vars={'prev_action': np.zeros((2, 1, action_dim))},
            name='p2_sym')
        dist = self.sess.run(
            dist_sym, feed_dict={obs_ph: [[obs.flatten()], [obs.flatten()]]})

        assert np.array_equal(dist['prob'], np.full((2, 1, action_dim), 0.5))

    def test_dist_info_sym_wrong_input(self):
        env = TfEnv(DummyDiscreteEnv(obs_dim=(1, ), action_dim=1))

        obs_ph = tf.placeholder(
            tf.float32, shape=(None, None, env.observation_space.flat_dim))

        with mock.patch(('garage.tf.policies.'
                         'categorical_gru_policy_with_model.GRUModel'),
                        new=SimpleGRUModel):
            policy = CategoricalGRUPolicyWithModel(
                env_spec=env.spec, state_include_action=True)

        policy.reset()
        obs = env.reset()

        policy.dist_info_sym(
            obs_var=obs_ph,
            state_info_vars={'prev_action': np.zeros((3, 1, 1))},
            name='p2_sym')
        # observation batch size = 2 but prev_action batch size = 3
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.sess.run(
                policy.model.networks['p2_sym'].input,
                feed_dict={obs_ph: [[obs.flatten()], [obs.flatten()]]})

    def test_invalid_env(self):
        env = TfEnv(DummyBoxEnv())
        with mock.patch(('garage.tf.policies.'
                         'categorical_gru_policy_with_model.GRUModel'),
                        new=SimpleGRUModel):
            with self.assertRaises(ValueError):
                CategoricalGRUPolicyWithModel(env_spec=env.spec)

    @params(
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    )
    @mock.patch('numpy.random.rand')
    def test_get_action_state_include_action(self, obs_dim, action_dim,
                                             hidden_dim, mock_rand):
        mock_rand.return_value = 0

        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))

        with mock.patch(('garage.tf.policies.'
                         'categorical_gru_policy_with_model.GRUModel'),
                        new=SimpleGRUModel):
            policy = CategoricalGRUPolicyWithModel(
                env_spec=env.spec, state_include_action=True)

        policy.reset()
        obs = env.reset()

        expected_prob = np.full(action_dim, 0.5)

        action, agent_info = policy.get_action(obs)
        assert env.action_space.contains(action)
        assert action == 0
        assert np.array_equal(agent_info['prob'], expected_prob)

        actions, agent_infos = policy.get_actions([obs])
        for action, prob in zip(actions, agent_infos['prob']):
            assert env.action_space.contains(action)
            assert action == 0
            assert np.array_equal(prob, expected_prob)

    @params(
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    )
    @mock.patch('numpy.random.rand')
    def test_get_action(self, obs_dim, action_dim, hidden_dim, mock_rand):
        mock_rand.return_value = 0

        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))

        with mock.patch(('garage.tf.policies.'
                         'categorical_gru_policy_with_model.GRUModel'),
                        new=SimpleGRUModel):
            policy = CategoricalGRUPolicyWithModel(
                env_spec=env.spec, state_include_action=False)

        policy.reset()
        obs = env.reset()

        expected_prob = np.full(action_dim, 0.5)

        action, agent_info = policy.get_action(obs)
        assert env.action_space.contains(action)
        assert action == 0
        assert np.array_equal(agent_info['prob'], expected_prob)

        actions, agent_infos = policy.get_actions([obs])
        for action, prob in zip(actions, agent_infos['prob']):
            assert env.action_space.contains(action)
            assert action == 0
            assert np.array_equal(prob, expected_prob)

    def test_is_pickleable(self):
        env = TfEnv(DummyDiscreteEnv(obs_dim=(1, ), action_dim=1))
        with mock.patch(('garage.tf.policies.'
                         'categorical_gru_policy_with_model.GRUModel'),
                        new=SimpleGRUModel):
            policy = CategoricalGRUPolicyWithModel(
                env_spec=env.spec, state_include_action=False)

        env.reset()
        obs = env.reset()

        with tf.variable_scope(
                'CategoricalGRUPolicyWithModel/prob_network', reuse=True):
            return_var = tf.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = self.sess.run(
            policy.model.outputs[0],
            feed_dict={policy.model.input: [[obs.flatten()], [obs.flatten()]]})

        p = pickle.dumps(policy)

        with tf.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(
                policy_pickled.model.outputs[0],
                feed_dict={
                    policy_pickled.model.input: [[obs.flatten()],
                                                 [obs.flatten()]]
                })
            assert np.array_equal(output1, output2)
