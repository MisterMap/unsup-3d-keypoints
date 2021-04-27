from kapture.io.features import get_descriptors_fullpath, get_keypoints_fullpath, \
    image_descriptors_from_file, image_keypoints_from_file
import kapture
import numpy as np


class KeypointMap(object):
    def __init__(self):
        self.points3d = None
        self.keypoints = None
        self.descriptors = None
        self.mask = None
        self.positions = None

    def load_from_kapture(self, kapture_data, minimal_observation_count=10):
        image_list = [filename for _, _, filename in kapture.flatten(kapture_data.records_camera)]
        descriptors = []
        keypoints = []
        points3d = []
        mask = []
        image_indexes = {}
        for i, image_path in enumerate(image_list):
            descriptors_full_path = get_descriptors_fullpath(kapture_data.kapture_path, image_path)
            descriptors.append(image_descriptors_from_file(descriptors_full_path, kapture_data.descriptors.dtype,
                                                           kapture_data.descriptors.dsize))
            keypoints_full_path = get_keypoints_fullpath(kapture_data.kapture_path, image_path)
            keypoints.append(image_keypoints_from_file(keypoints_full_path, kapture_data.keypoints.dtype,
                                                       kapture_data.keypoints.dsize))
            points3d.append(np.zeros((len(keypoints[i]), 3), dtype=np.float32))
            mask.append(np.zeros(len(keypoints[i]), dtype=np.bool))
            image_indexes[image_path] = i

        for point_index, observation in kapture_data.observations.items():
            if len(observation) > minimal_observation_count:
                for observation_image_name, image_keypoints_index in observation:
                    image_index = image_indexes[observation_image_name]
                    mask[image_index][image_keypoints_index] = True
                    points3d[image_index][image_keypoints_index] = kapture_data.points3d[point_index][:3]
        self.descriptors = descriptors
        self.keypoints = keypoints
        self.points3d = points3d
        self.mask = mask

    def get_descriptors(self):
        return np.concatenate(self.descriptors, axis=0)

    def get_mask(self):
        return np.concatenate(self.mask, axis=0)

    def get_points3d(self):
        return np.concatenate(self.points3d, axis=0)
