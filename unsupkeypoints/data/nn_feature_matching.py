import kapture
import kapture.core.Kapture
import kapture.core.Matches
from kapture.io.features import get_descriptors_fullpath, image_descriptors_from_file, get_matches_fullpath, \
    image_matches_to_file
from kapture_localization.matching.matching import MatchPairNnTorch
from tqdm import tqdm as tqdm


class NNFeatureMatching(object):
    def __init__(self, minimal_score=None, sequential_length=None):
        self._matcher = MatchPairNnTorch()
        self._minimal_score = minimal_score
        self._sequential_length = sequential_length

    def match_features(self, kapture_data):
        image_list = [filename for _, _, filename in kapture.flatten(kapture_data.records_camera)]
        descriptors = []
        descriptor_type = kapture_data.descriptors.dtype
        descriptor_size = kapture_data.descriptors.dsize
        for image_path in image_list:
            descriptors_full_path = get_descriptors_fullpath(kapture_data.kapture_path, image_path)
            descriptors.append(image_descriptors_from_file(descriptors_full_path, descriptor_type, descriptor_size))
        kapture_data.matches = kapture.Matches()
        if self._sequential_length is None:
            self._sequential_length = len(image_list)
        for i in tqdm(range(len(image_list))):
            for j in range(i + 1, min(len(image_list), i + self._sequential_length)):
                matches = self._matcher.match_descriptors(descriptors[i], descriptors[j])
                if self._minimal_score is not None:
                    mask = matches[:, 2] > self._minimal_score
                    matches = matches[mask]
                kapture_data.matches.add(image_list[i], image_list[j])
                matches_full_path = get_matches_fullpath((image_list[i], image_list[j]), kapture_data.kapture_path)
                image_matches_to_file(matches_full_path, matches)
