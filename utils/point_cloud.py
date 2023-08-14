import numpy as np
import cv2
import open3d.ml.torch as ml3d
import torch
from scipy.spatial.transform import Rotation
from typing import Optional
from scipy.spatial import cKDTree
import trimesh

def load_inseg(pth_ply):
    cloud_pd = trimesh.load(pth_ply, process=False)
    points_pd = cloud_pd.vertices
    segments_pd = cloud_pd.metadata['_ply_raw']['vertex']['data']['label'].flatten()

    return cloud_pd, points_pd, segments_pd

def load_obj(filename):
    with open(filename, 'r') as f:
        vertices = []
        faces = []
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]]
                faces.append(face)
        
    v = np.asarray(vertices)
    f = np.asarray(faces)
    assert v.shape[1] == f.shape[1]
    return v, f

def normalize_pc(pc, return_distances=False):
    pc_ = pc[:,:3]
    centroid = np.mean(pc_, axis=0)
    pc_ = pc_ - centroid
    m = np.max(np.sqrt(np.sum(pc_ ** 2, axis=1)))
    pc_ = pc_ / m
    if pc.shape[1] > 3:
        pc = np.concatenate((pc_, pc[:,3].reshape(-1,1)), axis=1)
    else:
        pc = pc_
    
    if return_distances:
        return pc, centroid, m
    else:
        return pc

def pcl_random_sample(point, npoint):
    N, D = point.shape

    if N < npoint:
        indices = np.random.choice(point.shape[0], npoint, replace=True)
    else:
        indices = np.random.choice(point.shape[0], npoint, replace=False)
    
    point = point[indices]
    return point

def pcl_farthest_sample(point, npoint, return_idxs = False):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N < npoint:
        indices = np.random.choice(point.shape[0], npoint)
        point = point[indices]
        return point

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]

    if return_idxs: return point, centroids.astype(np.int32)
    return point

def compute_pcl_overlap(source, target, threshold=1e-7):
    '''
    Compute overlap ratio from source point cloud to target point cloud
    '''
    points = torch.from_numpy(np.asarray(source)).to(torch.float64)
    queries = torch.from_numpy(np.asarray(target)).to(torch.float64)
    radii = torch.tensor([threshold]).tile(queries.size(0)).to(torch.float64)
    nsearch = ml3d.layers.RadiusSearch(return_distances=False)
    ans = nsearch(points, queries, radii)
    common_pts_idx_src = np.unique(ans[0].numpy())
    
    overlap_ratio = round(common_pts_idx_src.shape[0] / source.shape[0], 4)
    return overlap_ratio, common_pts_idx_src

def inverse_relative(pose1To2):
    pose2To1 = np.zeros((4, 4), dtype='float32')
    pose2To1[:3, :3] = np.transpose(pose1To2[:3, :3])
    pose2To1[:3, 3:4] = -np.dot(np.transpose(pose1To2[:3, :3]), pose1To2[:3, 3:4])
    pose2To1[3, 3] = 1
    return pose2To1

def get_visible_pts_from_cam_pose(scene_pts, cam_2_world_pose, intrinsic_info):
    '''
    Given a scene PCl, return the points visible in the given frame
    '''
    rvec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tvec = np.array([0.0, 0.0, 0.0])
    dist_coeffs = np.array([.0, .0, .0, .0, .0])

    world2CamPose = inverse_relative(cam_2_world_pose) # change to world2cam

    cam_pts_3d = np.concatenate((scene_pts, np.tile(np.array([1]), (scene_pts.shape[0], 1))), axis=1).dot(world2CamPose.T)
    cam_pts_3d = cam_pts_3d[..., :3]
    
    # Project Points to 2D
    out, _ = cv2.projectPoints(cam_pts_3d.reshape(-1, 1, 3), rvec, tvec, intrinsic_info['intrinsic_mat'], distCoeffs=dist_coeffs)
    out = out.reshape(-1, 2)

    # Check if pts within frame and if depth > 0
    out_x_mask = (out[..., 0] >= 0) & (out[..., 0] <= intrinsic_info['height'])
    out_y_mask = (out[..., 1] >= 0) & (out[..., 1] <= intrinsic_info['width'])
    depth_mask = cam_pts_3d[..., 2] > 0.0
    visible_mask = np.logical_and(depth_mask, np.logical_and(out_x_mask, out_y_mask))
    return visible_mask

def get_nearest_neighbor(
    q_points: np.ndarray,
    s_points: np.ndarray,
    return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1) #, n_jobs=-1)
    if return_index:
        return distances, indices
    else:
        return distances

def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points

def sample_faces(vertices, faces, n_samples=10**4):
  """
  Samples point cloud on the surface of the model defined as vectices and
  faces. This function uses vectorized operations so fast at the cost of some
  memory.

  Parameters:
    vertices  - n x 3 matrix
    faces     - n x 3 matrix
    n_samples - positive integer

  Return:
    vertices - point cloud

  Reference :
    [1] Barycentric coordinate system

    \begin{align}
      P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
    \end{align}
  """
  vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
  face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
  face_areas = face_areas / np.sum(face_areas)

  # Sample exactly n_samples. First, oversample points and remove redundant
  # Error fix by Yangyan (yangyan.lee@gmail.com) 2017-Aug-7
  n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
  floor_num = np.sum(n_samples_per_face) - n_samples
  if floor_num > 0:
    indices = np.where(n_samples_per_face > 0)[0]
    floor_indices = np.random.choice(indices, floor_num, replace=True)
    n_samples_per_face[floor_indices] -= 1

  n_samples = np.sum(n_samples_per_face)

  # Create a vector that contains the face indices
  sample_face_idx = np.zeros((n_samples, ), dtype=int)
  acc = 0
  for face_idx, _n_sample in enumerate(n_samples_per_face):
    sample_face_idx[acc: acc + _n_sample] = face_idx
    acc += _n_sample

  r = np.random.rand(n_samples, 2);
  A = vertices[faces[sample_face_idx, 0], :]
  B = vertices[faces[sample_face_idx, 1], :]
  C = vertices[faces[sample_face_idx, 2], :]
  P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
      np.sqrt(r[:,0:1]) * r[:,1:] * C
  return P