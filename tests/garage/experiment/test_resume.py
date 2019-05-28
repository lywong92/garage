import tempfile

import numpy as np

from garage.experiment import LocalRunner, SnapshotConfig
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase


def fixture_exp(snapshot_config):
    with LocalRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(
            name='policy', env_spec=env.spec, hidden_sizes=(8, 8))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            optimizer_args=dict(tf_optimizer_args=dict(learning_rate=0.01, )))

        runner.setup(algo, env)
        runner.train(n_epochs=5, batch_size=100)

        return policy.get_param_values()


class TestResume(TfGraphTestCase):
    def test_resume(self):
        temp_dir = tempfile.TemporaryDirectory()

        snapshot_config = SnapshotConfig(
            snapshot_dir=temp_dir.name, snapshot_mode='last', snapshot_gap=1)

        policy_params = fixture_exp(snapshot_config)
        self.tearDown()
        self.setUp()

        with LocalRunner(snapshot_config) as runner:
            args = runner.restore()
            assert np.isclose(
                runner.policy.get_param_values(),
                policy_params).all(), 'Policy parameters should persist'
            assert args.n_epochs == 5, (
                'Snapshot should save training parameters')
            assert args.start_epoch == 5, (
                'Last experiment should end at 5th iterations')

            batch_size = runner.train_args.batch_size
            n_epoch_cycles = runner.train_args.n_epoch_cycles

            runner.resume(
                n_epochs=10,
                plot=False,
                store_paths=True,
                pause_for_plot=False)

            self.assertEqual(runner.train_args.n_epochs, 10)
            self.assertEqual(runner.train_args.batch_size, batch_size)
            self.assertEqual(runner.train_args.n_epoch_cycles, n_epoch_cycles)
            self.assertEqual(runner.train_args.plot, False)
            self.assertEqual(runner.train_args.store_paths, True)
            self.assertEqual(runner.train_args.pause_for_plot, False)

        temp_dir.cleanup()
