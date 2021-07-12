import numpy as np
from .pnp_estimator import PnPEstimator


class ResultSaver(dict):
    def __init__(self):
        super().__init__()
        self._pnp_estimator = PnPEstimator()

    def save(self, output, batch):
        self.add("point3d", batch["point3d"])
        self.add("keypoint", batch["keypoint"])
        self.add("mask", batch["mask"])
        self.add("image_index", batch["image_index"])
        self.add("position", batch["position"])
        self.add("predicted_point3d", output)

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
            self["mask"],
            self["position"])
        reconstruction_position_errors, reconstruction_rotation_errors = self._pnp_estimator.calculate_position_errors(
            self["point3d"],
            self["keypoint"],
            self["image_index"],
            self["mask"],
            self["position"])
        return {
            "median_position_error": np.median(position_errors),
            "median_rotation_error": np.median(rotation_errors),
            "reconstruction_median_position_error": np.median(reconstruction_position_errors),
            "reconstruction_median_rotation_error": np.median(reconstruction_rotation_errors),
            "point_count": len(position_errors)
        }
