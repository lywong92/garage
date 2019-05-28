import base64
import unittest.mock

import cloudpickle

from garage.experiment import SnapshotConfig
from garage.experiment.experiment_wrapper import run_experiment


class TestExperimentWrapper(unittest.TestCase):
    @staticmethod
    def method_call(snapshot_config, variant_data):
        assert isinstance(snapshot_config, SnapshotConfig)
        assert variant_data is None

    def test_experiment_wrapper_method_call(self):
        data = base64.b64encode(
            cloudpickle.dumps(
                TestExperimentWrapper.method_call)).decode('utf-8')
        args = [
            '',
            '--args_data',
            data,
            '--use_cloudpickle',
            'True',
        ]
        run_experiment(args)
