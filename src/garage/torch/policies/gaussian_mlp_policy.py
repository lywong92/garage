import torch
from torch import nn

from garage.torch.modules import GaussianMLPModule
from garage.torch.policies import Policy


class GaussianMLPPolicy(nn.Module, Policy):
    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_output_nonlinearity=None,
                 std_parameterization='exp',
                 layer_normalization=False):
        nn.Module.__init__(self)
        Policy.__init__(self, env_spec)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        self._model = GaussianMLPModule(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            adaptive_std=adaptive_std,
            std_share_network=std_share_network,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_hidden_sizes=std_hidden_sizes,
            std_hidden_nonlinearity=std_hidden_nonlinearity,
            std_output_nonlinearity=std_output_nonlinearity,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization
        )

    def forward(self, inputs):
        return self._model(inputs)

    def get_action(self, observation):
        action_var, mean, log_std, _, __ = self.forward(observation)

        return (action_var[0].detach().numpy(),
                dict(
                   mean=mean[0].detach().numpy(),
                   log_std=log_std[0].detach().numpy()
                ))

    def get_actions(self, observations):
        action_var, mean, log_std, _, __ = self.forward(observations)

        return (action_var.detach().numpy(),
                dict(
                   mean=mean.detach().numpy(),
                   log_std=log_std.detach().numpy()
                ))
