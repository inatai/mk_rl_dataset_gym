import numpy as np
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])




class MyTrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, mode_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.mode_dim = mode_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, mdoes, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None, None

    def get_action(self, states, actions, rewards, modes, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
