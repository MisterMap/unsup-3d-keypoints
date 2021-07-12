import unittest
from unsupkeypoints.utils.math import *
import numpy as np


class TestMath(unittest.TestCase):
    def test_solve_pnp(self):
        points3d = np.array([[4., 4., 4.], [5., 4., 4.], [4., 5., 4.], [6., 6., 7.]], dtype=float)
        translation = np.array([1., 2., 3.])
        rotation = np.array([0.1, 0.2, 0.3])
        matrix_position = np.eye(4)
        matrix_position[:3, 3] = translation
        matrix_position[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
        camera_matrix = np.array([[525., 0, 320],
                                  [0, 525., 240],
                                  [0, 0, 1.]])
        invert_position = invert_positions(matrix_position[None])[0]
        local_point3d = (camera_matrix @ (invert_position[:3, :3] @ points3d.T + invert_position[:3, 3:4])).T
        keypoints = local_point3d[:, :2] / local_point3d[:, 2][:, None]

        distorsion_coefs = np.zeros(5)
        recovered_position = pnp_position(points3d, keypoints, camera_matrix, distorsion_coefs)
        recovered_translation = recovered_position[:3, 3]
        recovered_rotation = Rotation.from_matrix(recovered_position[:3, :3]).as_euler("xyz")
        self.assertAlmostEqual(np.linalg.norm(translation - recovered_translation), 0, delta=1e-3)
        self.assertAlmostEqual(np.linalg.norm(rotation - recovered_rotation), 0, delta=1e-3)
