class ResultSaver(object):
    def __init__(self):
        self._points3d = []
        self._keypoints = []
        self._positions = []

    def save(self, output, batch):
        pass

    def clear(self):
        self._points3d = []
        self._keypoints = []
        self._positions = []

    def get_metrics(self):
        pass
