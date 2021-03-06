import tensorflow as tf

from garage.tf.regressors import Regressor2
from tests.fixtures.models import SimpleMLPModel


class SimpleMLPRegressor(Regressor2):
    """Simple GaussianMLPRegressor for testing."""

    def __init__(self, input_shape, output_dim, name, *args, **kwargs):
        super().__init__(input_shape, output_dim, name)

        self.model = SimpleMLPModel(
            output_dim=self._output_dim, name='SimpleMLPModel')

        self._initialize()

    def _initialize(self):
        input_ph = tf.placeholder(
            tf.float32, shape=(None, ) + self._input_shape)
        with tf.variable_scope(self._name) as vs:
            self._variable_scope = vs
            self.model.build(input_ph)
        self.ys = None

    def fit(self, xs, ys):
        self.ys = ys

    def predict(self, xs):
        if self.ys is None:
            outputs = tf.get_default_session().run(
                self.model.networks['default'].outputs,
                feed_dict={self.model.networks['default'].input: xs})
            self.ys = outputs

        return self.ys

    def get_params_internal(self, *args, **kwargs):
        return self._variable_scope.trainable_variables()

    def __setstate__(self, state):
        """Object.__setstate__."""
        super().__setstate__(state)
        self._initialize()
