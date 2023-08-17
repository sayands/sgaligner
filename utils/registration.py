import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

import sys
sys.path.append('..')
from utils.point_cloud import apply_transform, get_nearest_neighbor

def compute_modified_chamfer_distance(src_points, ref_points, raw_points, est_transform, gt_transform):
    aligned_src_points = apply_transform(src_points, est_transform)
    chamfer_distance_p_q = get_nearest_neighbor(aligned_src_points, raw_points).mean()
    composed_transform = np.matmul(est_transform, np.linalg.inv(gt_transform))
    aligned_raw_points = apply_transform(raw_points, composed_transform)
    chamfer_distance_q_p = get_nearest_neighbor(ref_points, aligned_raw_points).mean()

    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
    return chamfer_distance

def compute_inlier_ratio(ref_corr_points, src_corr_points, transform, positive_radius=0.1):
    r"""Computing the inlier ratio between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    inlier_ratio = np.mean(residuals < positive_radius)    
    return inlier_ratio

def compute_registration_rmse(ref_points, src_points, transform):
    src_points = apply_transform(src_points, transform)
    rmse = np.sqrt(((ref_points-src_points) ** 2).sum() / src_points.shape[0])
    return rmse

def get_rotation_translation_from_transform(transform, inverse_trans=False):
    rotation = transform[:3, :3]
    
    if inverse_trans:
        translation = transform[3, :3]
    else:
        translation = transform[:3, 3]

    return rotation, translation

def compute_translation_mse_and_mae(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute anisotropic translation error (MSE and MAE)."""
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))

    return mse, mae

def compute_rotation_mse_and_mae(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute anisotropic rotation error (MSE and MAE)."""
    gt_euler_angles = Rotation.from_matrix(gt_rotation).as_euler('xyz', degrees=True)  # (3,)
    est_euler_angles = Rotation.from_matrix(est_rotation).as_euler('xyz', degrees=True)  # (3,)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    
    return mse, mae

def compute_transform_mse_and_mae(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute anisotropic rotation and translation error (MSE and MAE)."""
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.
    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)
    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)
    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre

def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.
    RTE = \lVert t - \bar{t} \rVert_2
    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)
    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)

def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray, inverse_trans=False):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.
    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)
    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform, inverse_trans)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    
    return rre, rte

def nn_correspondence(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

def compute_mosaicking_error(verts_pred, verts_gt, threshold=0.05):
    _, dist1 = nn_correspondence(verts_pred, verts_gt)
    _, dist2 = nn_correspondence(verts_gt, verts_pred)

    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recall = np.mean((dist1 < threshold).astype('float'))
    f1_score = 2 * precision * recall / (precision + recall)

    result_dict = {'prec' : precision, 'recall' : recall, 'acc' : np.mean(dist1), 'comp' : np.mean(dist2), 'fscore' : f1_score}
    return result_dict
