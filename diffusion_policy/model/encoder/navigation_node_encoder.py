import torch
import torch.nn as nn

from diffusion_policy.model.encoder.exploration_node_encoder import (
    ExplorationNodeEncoder,
)

# node feature layout: [rel_x, rel_y, utility, guidepost, occupancy, goal_dx, goal_dy]
GOAL_FEATURE_SLICE = slice(5, 7)


class NavigationNodeEncoder(ExplorationNodeEncoder):
    """Goal-fused graph encoder for navigation node features.

    The exploration encoder summarizes the belief graph from the current
    node's point of view only. For goal-conditioned navigation the region
    around the goal matters just as much, so this encoder additionally
    decodes the state of the node closest to the goal (smallest goal-offset
    feature norm) and fuses both views into one embedding. Inputs, output
    shape, and the forward signature are unchanged, so the diffusion policy
    and dataset format stay compatible.
    """

    def __init__(self, node_dim, embedding_dim):
        super().__init__(node_dim, embedding_dim)
        self.goal_fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def _goal_node_index(self, node_inputs, node_padding_mask):
        # goal offsets are zero on padded nodes; mask them out so the
        # argmin picks a real node
        goal_offsets = node_inputs[..., GOAL_FEATURE_SLICE]
        goal_distance = torch.norm(goal_offsets, dim=-1)  # (B, N)
        if node_padding_mask is not None:
            padding = node_padding_mask.squeeze(1).bool()  # (B, N)
            goal_distance = goal_distance.masked_fill(padding, float('inf'))
        return goal_distance.argmin(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        enhanced_node_feature = self.encode_graph(
            node_inputs, node_padding_mask, edge_mask
        )
        _, robot_belief_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask
        )

        goal_index = self._goal_node_index(node_inputs, node_padding_mask)
        _, goal_belief_feature = self.decode_state(
            enhanced_node_feature, goal_index, node_padding_mask
        )

        fused = self.goal_fusion(
            torch.cat((robot_belief_feature, goal_belief_feature), dim=-1)
        )
        return fused
