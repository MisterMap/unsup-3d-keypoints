import torch
import torch.nn as nn


class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, predicted_point3d, keypoint, point3d, position):
        keypoint_count = predicted_point3d.shape[0]
        loss = self.loss(predicted_point3d, point3d) / keypoint_count
        point_distance = torch.mean(torch.sqrt(torch.sum((predicted_point3d - point3d) ** 2, dim=1)))
        return {"loss": loss,
                "mean_dist": point_distance}
