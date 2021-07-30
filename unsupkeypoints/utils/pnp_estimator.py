from .math import pnp_position, calculate_errors
import numpy as np


class PnPEstimator(object):
    def __init__(self):
        self._camera_matrix = np.array([[525., 0, 320],
                                     [0, 525., 240],
                                     [0, 0, 1]])
        self._dist_coef = np.zeros(5)
        self._point_count_threshold = 5

    def calculate_position_errors(self, points3d, keypoint, image_index, ground_truth_positions):
        image_index_set = set(image_index)
        final_truth_positions = []
        final_predicted_positions = []
        for index in image_index_set:
            image_mask = image_index == index
            if np.count_nonzero(image_mask) >= self._point_count_threshold:
                image_points3d = points3d[image_mask]
                image_keypoints = keypoint[image_mask]
                final_truth_positions.append(ground_truth_positions[image_mask][0])
                recovered_position = pnp_position(image_points3d, image_keypoints, self._camera_matrix, self._dist_coef)
                final_predicted_positions.append(recovered_position)
        if len(final_truth_positions) == 0:
            return np.zeros(1), np.zeros(1)
        return calculate_errors(np.array(final_predicted_positions), np.array(final_truth_positions), False)
