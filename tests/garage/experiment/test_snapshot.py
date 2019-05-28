import os.path as osp
import tempfile

from dowel import logger
import joblib
import tensorflow as tf

from garage.experiment import LocalRunner, SnapshotConfig
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler.utils import rollout
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.logger import NullOutput


class TestSnapshot(TfGraphTestCase):
    verifyItrs = 3

    @classmethod
    def reset_tf(cls):
        if tf.get_default_session():
            tf.get_default_session().__exit__(None, None, None)
        tf.reset_default_graph()

    @classmethod
    def setUpClass(cls):
        cls.reset_tf()
        logger.add_output(NullOutput())

    @classmethod
    def tearDownClass(cls):
        logger.remove_all()

    def test_snapshot(self):
        temp_dir = tempfile.TemporaryDirectory()

        snapshot_config = SnapshotConfig(
            snapshot_dir=temp_dir.name, snapshot_mode='all', snapshot_gap=1)
        with LocalRunner(snapshot_config=snapshot_config) as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(
                name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                max_kl_step=0.01)

            runner.setup(algo, env)
            runner.train(n_epochs=self.verifyItrs, batch_size=4000)

            env.close()

        # Read snapshot from self.log_dir
        # Test the presence and integrity of policy and env
        for i in range(0, self.verifyItrs):
            self.reset_tf()
            with LocalRunner():
                snapshot = joblib.load(
                    osp.join(temp_dir.name, 'itr_{}.pkl'.format(i)))

                env = snapshot['env']
                algo = snapshot['algo']
                assert env
                assert algo
                assert algo.policy

                rollout(env, algo.policy, animated=False)

        temp_dir.cleanup()
