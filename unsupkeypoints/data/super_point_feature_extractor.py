import os.path

import cv2
from PIL import Image
from kapture.io.csv import *
from kapture.io.features import *
from kapture.io.records import *
from tqdm import tqdm as tqdm

from ..features.super_point_frontend import SuperPointFrontend


class SuperPointFeatureExtractor(object):
    def __init__(self, weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=True):
        self._super_point_frontend = SuperPointFrontend(weights_path=weights_path,
                                                        nms_dist=nms_dist,
                                                        conf_thresh=conf_thresh,
                                                        nn_thresh=nn_thresh,
                                                        cuda=cuda)

    def get_keypoints(self, image_path):
        image = np.asarray(Image.open(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = (image / 255.).astype(np.float32)

        keypoints, descriptors, heatmap = self._super_point_frontend.run(image)
        keypoints = keypoints[:2, :].T
        descriptors = descriptors.T
        return keypoints, descriptors

    def extract_features(self, kapture_data):
        image_list = [filename for _, _, filename in kapture.flatten(kapture_data.records_camera)]
        for i in tqdm(range(len(image_list))):
            image_path = os.path.join(kapture_data.image_path, image_list[i])
            keypoints, descriptors = self.get_keypoints(image_path)
            if i == 0:
                kapture_data.keypoints = kapture.Keypoints('superpoint', keypoints.dtype.type, keypoints.shape[1])
                kapture_data.descriptors = kapture.Descriptors('superpoint', descriptors.dtype.type, descriptors.shape[1])

                keypoints_config_absolute_path = get_csv_fullpath(kapture.Keypoints, kapture_data.kapture_path)
                descriptors_config_absolute_path = get_csv_fullpath(kapture.Descriptors, kapture_data.kapture_path)

                keypoints_to_file(keypoints_config_absolute_path, kapture_data.keypoints)
                descriptors_to_file(descriptors_config_absolute_path, kapture_data.descriptors)

            kapture_data.keypoints.add(image_list[i])
            kapture_data.descriptors.add(image_list[i])
            keypoints_full_path = get_keypoints_fullpath(kapture_data.kapture_path, image_list[i])
            image_keypoints_to_file(keypoints_full_path, keypoints)
            descriptors_full_path = get_descriptors_fullpath(kapture_data.kapture_path, image_list[i])
            image_descriptors_to_file(descriptors_full_path, descriptors)
