from torch.utils.data import Dataset

from .kapture_data import KaptureData
from ..features.keypoint_map import KeypointMap


class DescriptorDataset(Dataset):
    def __init__(self, path, minimal_observation_count=10):
        kapture_data = KaptureData.load_from(path, path, "")
        self._keypoint_map = KeypointMap()
        self._keypoint_map.load_from_kapture(kapture_data, minimal_observation_count)

    def __getitem__(self, item):
        image_index = self._keypoint_map.masked_image_index_list[item]
        keypoint_index = self._keypoint_map.masked_keypoint_index_list[item]

        return {"descriptor": self._keypoint_map.descriptors[image_index][keypoint_index],
                "point3d": self._keypoint_map.points3d[image_index][keypoint_index],
                "keypoint": self._keypoint_map.keypoints[image_index][keypoint_index],
                "position": self._keypoint_map.positions[image_index],
                "image_index": image_index}

    def __len__(self):
        return len(self._keypoint_map.masked_keypoint_index_list)
