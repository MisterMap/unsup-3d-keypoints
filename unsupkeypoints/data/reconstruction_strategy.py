class ReconstructionStrategy(object):
    def __init__(self, feature_extractor, feature_matching, reconstructor_backend):
        self._feature_extractor = feature_extractor
        self._feature_matching = feature_matching
        self._reconstructor_backend = reconstructor_backend

    def make_reconstruction(self, data):
        self._feature_extractor.extract_features(data)
        self._feature_matching.match_features(data)
        self._reconstructor_backend.reconstruct(data)
        data.save()
