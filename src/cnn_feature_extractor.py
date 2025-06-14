import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, num_actions=6, embed_dim=8):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        """ Compute the output size of the CNN """
        with torch.no_grad():
            sample_lidar = torch.zeros((1, 1, 360), dtype=torch.float32)
            cnn_output_dim = self.cnn(sample_lidar).shape[1]

        """ Embed the integer action as a dense vector """
        """ Add 1 to num_actions to include a special embedding for "no previous action" (-1) """
        self.embedding = nn.Embedding(num_actions + 1, embed_dim)
        self.num_actions = num_actions

        """ Total is the CNN output dimension + embedding for the previous action """
        total_input_dim = cnn_output_dim + embed_dim

        self.linear = nn.Sequential(
            nn.Linear(total_input_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Assuming observation array is:
        observations[0:360] --> LiDAR
        observations[360:] --> extra data (last action)
        """
        lidar = observations[:, :360]
        prev_action = observations[:, 360].long()

        """ CNN expects [batch_size, 1, 360] """
        x = self.cnn(lidar.unsqueeze(1))

        """
        Handle -1 (no previous action) by mapping it to the last embedding index
        Invalid action: -1 -> embedding index: 6 (num_actions)
        """
        prev_action_embedded_idx = torch.where(
            prev_action == -1, self.num_actions, prev_action
        )

        """ Clamp any other invalid values """
        prev_action_embedded_idx = torch.clamp(
            prev_action_embedded_idx, 0, self.num_actions
        )

        embedded = self.embedding(prev_action_embedded_idx)

        """ Concatenate CNN features with embedded previous action """
        combined = torch.cat([x, embedded], dim=1)

        """ Pass through the linear layer to get the final features """
        return self.linear(combined)
