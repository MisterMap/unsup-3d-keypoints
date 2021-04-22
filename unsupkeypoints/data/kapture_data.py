import kapture
from kapture import Kapture
from kapture.io.csv import kapture_from_dir, kapture_to_dir


class KaptureData(Kapture):
    def __init__(self, kapture_path, image_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kapture_path = kapture_path
        self.image_path = image_path

    @staticmethod
    def load_from(load_path, kapture_path, image_path):
        kapture_data = kapture_from_dir(load_path)
        return KaptureData(kapture_path, image_path, **kapture_data.as_dict(True))

    def save(self):
        kapture_to_dir(self.kapture_path, self)

    @property
    def image_names(self):
        image_list = [filename for _, _, filename in kapture.flatten(self.records_camera)]
        return {i + 1: x for i, x in enumerate(image_list)}
