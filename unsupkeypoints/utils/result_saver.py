import numpy as np
from .pnp_estimator import PnPEstimator


class ResultSaver(dict):
    def __init__(self):
        super().__init__()
        self._pnp_estimator = PnPEstimator()

    def save(self, result):
        for key, value in result.items():
            self.add(key, value)

    def add(self, key, tensor):
        tensor = tensor.detach().cpu().numpy()
        if key not in self.keys():
            self[key] = tensor
            return

        self[key] = np.concatenate([self[key], tensor], axis=0)

    def get_metrics(self):
        position_errors, rotation_errors = self._pnp_estimator.calculate_position_errors(
            self["predicted_point3d"],
            self["keypoint"],
            self["image_index"],
            self["position"])
        return {
            "median_position_error": np.median(position_errors),
            "median_rotation_error": np.median(rotation_errors),
            "point_count": len(position_errors)
        }
