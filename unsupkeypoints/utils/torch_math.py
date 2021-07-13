import torch
import math


def quaternion_angular_error(q1, q2):
    dot = torch.sum(q1 * q2, dim=1)
    d = torch.abs(dot)
    d = torch.clamp(d, -1, 1)
    theta = 2 * torch.acos(d) * 180 / math.pi
    return theta


def inverse_pose_matrix(matrix):
    result = torch.zeros_like(matrix)
    rotation_part = matrix[:, :3, :3]
    translation_part = matrix[:, :3, 3]
    rotation_part_transposed = torch.transpose(rotation_part, 1, 2)
    result[:, :3, :3] = rotation_part_transposed
    result[:, :3, 3] = -torch.bmm(rotation_part_transposed, translation_part[:, :, None])[:, :, 0]
    result[:, 3, 3] = 1
    return result
