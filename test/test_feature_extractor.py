from kapture.io.features import keypoints_check_dir, descriptors_check_dir

from unsupkeypoints.data.d2_net_feature_extractor import D2NetFeatureExtractor
from unsupkeypoints.data.kapture_data import KaptureData
from unsupkeypoints.data.nn_feature_matching import NNFeatureMatching
from unsupkeypoints.data.colmap_reconstructor import ColmapReconstructor

model_file = "/home/mikhail/research/d2-net/models/d2_ots.pth"
feature_extractor = D2NetFeatureExtractor(model_file=model_file, mininal_score=14)
kapture_path = "data/test_feature_extractor"
image_path = "/home/mikhail/research/unsup-3d-keypoints/data/kapture/7scenes/fire/mapping/sensors/records_data"
kapture_data = KaptureData.load_from("data/input_kapture_data", kapture_path, image_path)

feature_extractor.extract_features(kapture_data)

feature_matching = NNFeatureMatching()
feature_matching.match_features(kapture_data)

if not keypoints_check_dir(kapture_data.keypoints, kapture_path) or \
        not descriptors_check_dir(kapture_data.descriptors, kapture_path):
    print('local feature extraction ended successfully but not all files were saved')

colmap_reconstructor = ColmapReconstructor("data/tmp/colmap", "/usr/local/bin/colmap")
colmap_reconstructor.reconstruct(kapture_data)
kapture_data.save()