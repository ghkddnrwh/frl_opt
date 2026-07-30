from typing import List, Type, Tuple
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
from stable_baselines3.common.policies import BaseModel


class DiscreteProtester(BaseModel):
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.protester_network: nn.Module = None
        protester_net_list = create_mlp(features_dim, 1, net_arch, activation_fn)
        protester_net = nn.Sequential(*protester_net_list)
        self.add_module(f"pf", protester_net)
        self.protester_network = protester_net

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        # with th.set_grad_enabled(not self.share_features_extractor):
        features = self.extract_features(obs, self.features_extractor)
        return self.protester_network(features)
    

class Protester(BaseModel):
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        
        self.protester_network: nn.Module = None
        protester_net_list = create_mlp(features_dim, 1, net_arch, activation_fn)
        protester_net = nn.Sequential(*protester_net_list)
        self.add_module(f"pf", protester_net)
        self.protester_network = protester_net

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return self.protester_network(features)
