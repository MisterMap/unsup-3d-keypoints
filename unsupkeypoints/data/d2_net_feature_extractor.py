import os.path

import torch
from PIL import Image
from kapture.io.csv import *
from kapture.io.features import *
from kapture.io.records import *
from model_test import D2Net
from pyramid import process_multiscale
from tqdm import tqdm as tqdm
from utils import preprocess_image


class D2NetFeatureExtractor(object):
    def __init__(self, model_file, minimal_score=None):
        self._model = D2Net(
            model_file=model_file,
            use_relu=True,
            use_cuda=True
        )
        self._minimal_score = minimal_score

    def get_keypoints(self, image_path):
        image = np.asarray(Image.open(image_path))
        image_tensor = preprocess_image(image, preprocessing="torch")[None]
        image_tensor = torch.tensor(image_tensor).float().cuda()
        with torch.no_grad():
            keypoints, scores, descriptors = process_multiscale(image_tensor, self._model, scales=[1])
        keypoints = keypoints[:, :2][:, ::-1]
        if self._minimal_score is not None:
            mask = scores > self._minimal_score
            keypoints = keypoints[mask]
            descriptors = descriptors[mask]
        return keypoints, descriptors

    def extract_features(self, kapture_data):
        image_list = [filename for _, _, filename in kapture.flatten(kapture_data.records_camera)]
        for i in tqdm(range(len(image_list))):
            image_path = os.path.join(kapture_data.image_path, image_list[i])
            keypoints, descriptors = self.get_keypoints(image_path)
            if i == 0:
                kapture_data.keypoints = kapture.Keypoints('d2net', keypoints.dtype.type, keypoints.shape[1])
                kapture_data.descriptors = kapture.Descriptors('d2net', descriptors.dtype.type, descriptors.shape[1])

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
