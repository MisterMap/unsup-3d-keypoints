from kapture.io.features import get_descriptors_fullpath, get_keypoints_fullpath, \
    image_descriptors_from_file, image_keypoints_from_file
import kapture
import numpy as np
import quaternion


class KeypointMap(object):
    def __init__(self, descriptor_name="d2net"):
        self.points3d = None
        self.keypoints = None
        self.descriptors = None
        self.mask = None
        self.positions = None
        self.image_index_list = None
        self.keypoint_index_list = None
        self.image_index_from_image_name = None
        self.keypoint_count = 0
        self._descriptor_name = descriptor_name

    def load_from_kapture(self, kapture_data, minimal_observation_count=10):
        image_list = [filename for _, _, filename in kapture.flatten(kapture_data.records_camera)]
        descriptors = []
        keypoints = []
        points3d = []
        mask = []
        image_indexes = {}
        image_index_list = []
        keypoint_index_list = []
        self.keypoint_count = 0
        for i, image_path in enumerate(image_list):
            descriptors_full_path = get_descriptors_fullpath(self._descriptor_name, kapture_data.kapture_path,
                                                             image_path)
            kapture_descriptors = kapture_data.descriptors[self._descriptor_name]
            descriptors.append(image_descriptors_from_file(descriptors_full_path, kapture_descriptors.dtype,
                                                           kapture_descriptors.dsize))
            keypoints_full_path = get_keypoints_fullpath(self._descriptor_name, kapture_data.kapture_path, image_path)
            kapture_keypoints = kapture_data.keypoints[self._descriptor_name]
            keypoints.append(image_keypoints_from_file(keypoints_full_path, kapture_keypoints.dtype,
                                                       kapture_keypoints.dsize))
            point_count = len(keypoints[i])
            points3d.append(np.zeros((point_count, 3), dtype=np.float32))
            mask.append(np.zeros(point_count, dtype=np.bool))
            image_indexes[image_path] = i
            image_index_list.extend([i] * point_count)
            keypoint_index_list.extend(range(point_count))
            self.keypoint_count += point_count

        for point_index, observation in kapture_data.observations.items():
            if len(observation[self._descriptor_name]) > minimal_observation_count:
                for observation_image_name, image_keypoints_index in observation[self._descriptor_name]:
                    image_index = image_indexes[observation_image_name]
                    mask[image_index][image_keypoints_index] = True
                    points3d[image_index][image_keypoints_index] = kapture_data.points3d[point_index][:3]
        self.descriptors = descriptors
        self.keypoints = keypoints
        self.points3d = points3d
        self.mask = mask
        self.image_index_list = image_index_list
        self.keypoint_index_list = keypoint_index_list
        self.image_index_from_image_name = image_indexes
        self.load_trajectory(kapture_data)

    def load_trajectory(self, kapture_data):
        trajectory = np.zeros((len(self.image_index_from_image_name), 4, 4), dtype=np.float32)
        for timestamp, camera_id, image_name in kapture.flatten(kapture_data.records_camera):
            image_index = self.image_index_from_image_name[image_name]
            position = kapture_data.trajectories[timestamp][camera_id]
            position = self.matrix_from_position(position)
            trajectory[image_index] = position
        self.positions = trajectory

    @staticmethod
    def matrix_from_position(position):
        matrix = np.zeros((4, 4), dtype=np.float32)
        rotation = quaternion.as_rotation_matrix(position.r)
        matrix[:3, :3] = rotation.T
        matrix[:3, 3] = -rotation.T @ np.array(position.t[:, 0])
        matrix[3, 3] = 1
        return matrix

    def get_descriptors(self):
        return np.concatenate(self.descriptors, axis=0)

    def get_mask(self):
        return np.concatenate(self.mask, axis=0)

    def get_points3d(self):
        return np.concatenate(self.points3d, axis=0)
