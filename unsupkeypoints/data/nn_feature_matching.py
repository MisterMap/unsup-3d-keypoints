import kapture
import kapture.core.Kapture
import kapture.core.Matches
from kapture.io.features import get_descriptors_fullpath, image_descriptors_from_file, get_matches_fullpath, \
    image_matches_to_file
from kapture_localization.matching.matching import MatchPairNnTorch


class NNFeatureMatching(object):
    def __init__(self):
        self._matcher = MatchPairNnTorch()

    def match_features(self, kapture_data):
        image_list = [filename for _, _, filename in kapture.flatten(kapture_data.records_camera)]
        descriptors = []
        descriptor_type = kapture_data.descriptors.dtype
        descriptor_size = kapture_data.descriptors.dsize
        for image_path in image_list:
            descriptors_full_path = get_descriptors_fullpath(kapture_data.kapture_path, image_path)
            descriptors.append(image_descriptors_from_file(descriptors_full_path, descriptor_type, descriptor_size))
        kapture_data.matches = kapture.Matches()
        for i in range(len(image_list)):
            for j in range(i + 1, len(image_list)):
                matches = self._matcher.match_descriptors(descriptors[i], descriptors[j])
                kapture_data.matches.add(image_list[i], image_list[j])
                matches_full_path = get_matches_fullpath((image_list[i], image_list[j]), kapture_data.kapture_path)
                image_matches_to_file(matches_full_path, matches)
