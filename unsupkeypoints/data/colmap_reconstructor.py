import os
import os.path as path

import kapture.converter.colmap.database_extra as database_extra
from kapture.converter.colmap.database import COLMAPDatabase
from kapture.converter.colmap.import_colmap_reconstruction import import_from_colmap_points3d_txt
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.utils.paths import safe_remove_file, safe_remove_any_path
from kapture_localization.colmap.colmap_command import run_point_triangulator, run_model_converter


class ColmapReconstructor(object):
    def __init__(self, colmap_path, colmap_binary, point_triangulator_options=None):
        self._colmap_path = colmap_path
        self._colmap_binary = colmap_binary
        self._point_triangulator_options = point_triangulator_options

    def reconstruct(self, kapture_data):
        os.makedirs(self._colmap_path, exist_ok=True)

        if not (kapture_data.records_camera and kapture_data.sensors and
                kapture_data.keypoints and kapture_data.matches and kapture_data.trajectories):
            raise ValueError('records_camera, sensors, keypoints, matches, trajectories are mandatory')

        # Set fixed name for COLMAP database
        colmap_db_path = path.join(self._colmap_path, 'colmap.db')
        reconstruction_path = path.join(self._colmap_path, "reconstruction")
        priors_txt_path = path.join(self._colmap_path, "priors_for_reconstruction")

        safe_remove_file(colmap_db_path, True)
        safe_remove_any_path(reconstruction_path, True)
        safe_remove_any_path(priors_txt_path, True)
        os.makedirs(reconstruction_path, exist_ok=True)

        # COLMAP does not fully support rigs.
        if kapture_data.rigs is not None and kapture_data.trajectories is not None:
            # make sure, rigs are not used in trajectories.
            rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)
            kapture_data.rigs.clear()

        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        database_extra.kapture_to_colmap(kapture_data, kapture_data.kapture_path, colmap_db,
                                         export_two_view_geometry=True)
        colmap_db.close()

        os.makedirs(priors_txt_path, exist_ok=True)

        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        database_extra.generate_priors_for_reconstruction(kapture_data, colmap_db, priors_txt_path)
        colmap_db.close()

        # Point triangulator
        reconstruction_path = path.join(self._colmap_path, "reconstruction")
        os.makedirs(reconstruction_path, exist_ok=True)
        run_point_triangulator(
            self._colmap_binary,
            colmap_db_path,
            kapture_data.image_path,
            priors_txt_path,
            reconstruction_path,
            self._point_triangulator_options
        )
        run_model_converter(
            self._colmap_binary,
            reconstruction_path,
            reconstruction_path
        )
        points3d, observations = import_from_colmap_points3d_txt(os.path.join(reconstruction_path, "points3D.txt"),
                                                                 kapture_data.image_names)
        kapture_data.observations = observations
        kapture_data.points3d = points3d
