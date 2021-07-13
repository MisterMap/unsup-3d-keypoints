import torch
import torch.nn as nn
from ..utils.torch_math import inverse_pose_matrix


class RGBandModelReprojectionLoss(nn.Module):
    def __init__(self, camera_matrix=None, minimal_depth=0.1, maximal_distance=0.2, maximal_reprojection_loss=1000,
                 robust_maximal_reprojection_loss=100, distance_coef=500):
        if camera_matrix is None:
            camera_matrix = torch.tensor([[525., 0, 320],
                                             [0, 525., 240],
                                             [0, 0, 1.]])
        self._camera_matrix = camera_matrix
        self._minimal_depth = minimal_depth
        self._maximal_distance = maximal_distance
        self._maximal_reprojection_loss = maximal_reprojection_loss
        self._robust_maximal_reprojection_loss = robust_maximal_reprojection_loss
        self._distance_coef = distance_coef
        super().__init__()

    def forward(self, predicted_points3d, keypoints, truth_points3d, positions, mask):
        predicted_points3d = predicted_points3d[mask]
        keypoints = keypoints[mask]
        positions = positions[mask]
        truth_points3d = truth_points3d[mask]
        keypoint_count = predicted_points3d.shape[0]
        inverted_positions = inverse_pose_matrix(positions)
        transformed_predicted_points3d = torch.bmm(
            inverted_positions[:, :3, :3], predicted_points3d[:, :, None])[:, :, 0] + inverted_positions[:, :3, 3]
        camera_matrix = torch.repeat_interleave(self._camera_matrix[None], keypoint_count, dim=0).to(keypoints.device)
        predicted_keypoints = torch.bmm(camera_matrix, transformed_predicted_points3d[:, :, None])[:, :, 0]

        predicted_keypoints = predicted_keypoints[:, :2] / predicted_points3d[:, 2:3]
        reprojection_losses = torch.norm(predicted_keypoints - keypoints, p=2, dim=1)
        point_distances = torch.norm(predicted_points3d - truth_points3d, p=2, dim=1)

        good_point_mask = self._filter_points(transformed_predicted_points3d, reprojection_losses, point_distances)
        loss = torch.sum(torch.where(good_point_mask, self._robust_reprojection_loss(reprojection_losses),
                                     point_distances * self._distance_coef)) / keypoint_count
        good_point_ratio = torch.count_nonzero(good_point_mask) / keypoint_count
        return {
            "loss": loss,
            "reprojection_loss": torch.mean(reprojection_losses),
            "distance_loss": torch.mean(point_distances),
            "good_point_ratio": good_point_ratio
        }

    def _filter_points(self, transformed_points3d: torch.Tensor, reprojection_losses: torch.Tensor,
                       point_distances: torch.Tensor) -> torch.Tensor:
        mask1 = transformed_points3d[:, 2] > self._minimal_depth
        mask2 = point_distances < self._maximal_distance
        mask3 = reprojection_losses < self._maximal_reprojection_loss
        return mask1 & mask2 & mask3

    def _robust_reprojection_loss(self, reprojection_loss: torch.Tensor):
        robust_reprojection_loss = torch.sqrt(torch.abs(reprojection_loss) * self._robust_maximal_reprojection_loss)
        return torch.where(reprojection_loss < self._robust_maximal_reprojection_loss, reprojection_loss,
                           robust_reprojection_loss)
