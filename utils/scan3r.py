import os.path as osp
import numpy as np
import json
from glob import glob
from plyfile import PlyData, PlyElement

def get_scan_ids(dirname, split):
    filepath = osp.join(dirname, '{}_scans.txt'.format(split))
    scan_ids = np.genfromtxt(filepath, dtype = str)
    return scan_ids

def read_labels(plydata):
    data = plydata.metadata['_ply_raw']['vertex']['data']
    try:
        labels = data['objectId']
    except:
        labels = data['label']
    return labels

def load_intrinsics(data_dir, scan_id, type='color'):
    '''
    Load 3RScan intrinsic information
    '''
    info_path = osp.join(data_dir, scan_id, 'sequence', '_info.txt')

    width_search_string = 'm_colorWidth' if type == 'color' else 'm_depthWidth'
    height_search_string = 'm_colorHeight' if type == 'color' else 'm_depthHeight'
    calibration_search_string = 'm_calibrationColorIntrinsic' if type == 'color' else 'm_calibrationDepthIntrinsic'

    with open(info_path) as f:
        lines = f.readlines()
    
    for line in lines:
        if line.find(height_search_string) >= 0:
            intrinsic_height = line[line.find("= ") + 2 :]
        
        elif line.find(width_search_string) >= 0:
            intrinsic_width = line[line.find("= ") + 2 :]
        
        elif line.find(calibration_search_string) >= 0:
            intrinsic_mat = line[line.find("= ") + 2 :].split(" ")

            intrinsic_fx = intrinsic_mat[0]
            intrinsic_cx = intrinsic_mat[2]
            intrinsic_fy = intrinsic_mat[5]
            intrinsic_cy = intrinsic_mat[6]

            intrinsic_mat = np.array([[intrinsic_fx, 0, intrinsic_cx],
                                    [0, intrinsic_fy, intrinsic_cy],
                                    [0, 0, 1]])
            intrinsic_mat = intrinsic_mat.astype(np.float32)
    intrinsics = {'width' : float(intrinsic_width), 'height' : float(intrinsic_height), 
                  'intrinsic_mat' : intrinsic_mat}
    
    return intrinsics

def load_ply_data(data_dir, scan_id, label_file_name):
    filename_in = osp.join(data_dir, scan_id, label_file_name)
    file = open(filename_in, 'rb')
    ply_data = PlyData.read(file)
    file.close()
    return ply_data

def load_pose(data_dir, scan_id, frame_id):
    pose_path = osp.join(data_dir, scan_id, 'sequence', 'frame-{}.pose.txt'.format(frame_id))
    pose = np.genfromtxt(pose_path)
    return pose

def load_all_poses(data_dir, scan_id, frame_idxs):
    frame_poses = []
    for frame_idx in frame_idxs:
        frame_pose = load_pose(data_dir, scan_id, frame_idx)
        frame_poses.append(frame_pose)
    frame_poses = np.array(frame_poses)

    return frame_poses

def load_frame_idxs(data_dir, scan_id, skip=None):
    num_frames = len(glob(osp.join(data_dir, scan_id, 'sequence', '*.jpg')))

    if skip is None:
        frame_idxs = ['{:06d}'.format(frame_idx) for frame_idx in range(0, num_frames)]
    else:
        frame_idxs = ['{:06d}'.format(frame_idx) for frame_idx in range(0, num_frames, skip)]

    return frame_idxs

def read_transform_mat(filename):
    rescan2ref = {}
    with open(filename , "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.matrix(scans["transform"]).reshape(4,4)
    return rescan2ref

def load_plydata_npy(file_path, obj_ids = None, return_ply_data = False):
    ply_data = np.load(file_path)
    points =  np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))

    if obj_ids is not None:
        if type(obj_ids) == np.ndarray:
            obj_ids_pc = ply_data['objectId']
            obj_ids_pc_mask = np.isin(obj_ids_pc, obj_ids)
            points = points[np.where(obj_ids_pc_mask == True)[0]]
        else:
            obj_ids_pc = ply_data['objectId']
            points = points[np.where(obj_ids_pc == obj_ids)[0]]
    
    if return_ply_data: return points, ply_data
    else: return points

def find_cam_centers(frame_idxs, frame_poses):
    cam_centers = []

    for idx in range(len(frame_idxs)):
        cam_2_world_pose = frame_poses[idx]
        frame_pose = np.linalg.inv(cam_2_world_pose) # world To Cam
        frame_rot = frame_pose[:3, :3]
        frame_trans = frame_pose[:3, 3] * 1000.0
        cam_center = -np.matmul(np.transpose(frame_rot), frame_trans)
        cam_centers.append(cam_center / 1000.0)

    cam_centers = np.array(cam_centers).reshape((-1, 3))
    return cam_centers

def create_ply_data_predicted(ply_data, visible_pts_idx):
    x = ply_data['vertex']['x'][visible_pts_idx]
    y = ply_data['vertex']['y'][visible_pts_idx]
    z = ply_data['vertex']['z'][visible_pts_idx]
    object_id = ply_data['vertex']['label'][visible_pts_idx]

    vertices = np.empty(len(visible_pts_idx), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('objectId', 'h')])
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')
    vertices['objectId'] = object_id.astype('h')

    return vertices, object_id

def create_ply_data(ply_data, visible_pts_idx):
    x = ply_data['vertex']['x'][visible_pts_idx]
    y = ply_data['vertex']['y'][visible_pts_idx]
    z = ply_data['vertex']['z'][visible_pts_idx]
    red = ply_data['vertex']['red'][visible_pts_idx]
    green = ply_data['vertex']['green'][visible_pts_idx]
    blue = ply_data['vertex']['blue'][visible_pts_idx]
    object_id = ply_data['vertex']['objectId'][visible_pts_idx]
    global_id = ply_data['vertex']['globalId'][visible_pts_idx]
    nyu40_id = ply_data['vertex']['NYU40'][visible_pts_idx]
    eigen13_id = ply_data['vertex']['Eigen13'][visible_pts_idx]
    rio27_id = ply_data['vertex']['RIO27'][visible_pts_idx]

    vertices = np.empty(len(visible_pts_idx), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                                     ('objectId', 'h'), ('globalId', 'h'), ('NYU40', 'u1'), ('Eigen13', 'u1'), ('RIO27', 'u1')])
    
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')
    vertices['red'] = red.astype('u1')
    vertices['green'] = green.astype('u1')
    vertices['blue'] = blue.astype('u1')
    vertices['objectId'] = object_id.astype('h')
    vertices['globalId'] = global_id.astype('h')
    vertices['NYU40'] = nyu40_id.astype('u1')
    vertices['Eigen13'] = eigen13_id.astype('u1')
    vertices['RIO27'] = rio27_id.astype('u1')

    return vertices, object_id

# def remove_ceiling(self, points):
#         points_mask = points[..., 2] < np.max(points[..., 2]) - 1
#         points = points[points_mask]
#         return points
    
#     def create_vis(self):
#         vis = o3d.visualization.Visualizer()
#         vis.create_window(window_name='test', width=1280, height=840, left=0, top=0, visible=True)
#         vis.get_render_option().light_on = False
#         vis.get_render_option().line_width = 100.0

#         ctr = vis.get_view_control()
#         camera_param = ctr.convert_to_pinhole_camera_parameters()
#         camera_param.extrinsic = np.array([[ 0.61093874,  0.79161489, -0.00998646,  0.07131896],
#                                         [ 0.78591277, -0.60796109, -0.11280264,  0.56497598],
#                                         [-0.09536763,  0.06106702, -0.99356723,  6.99977745],
#                                         [ 0.        ,  0.        , 0.         , 1.        ]])
#         ctr.convert_from_pinhole_camera_parameters(camera_param)
#         return vis
    
#     def create_line_mesh_geoms(self, line_set, color):
#         lines = np.asarray(line_set.lines).tolist()
#         points = np.asarray(line_set.points)
#         line_mesh = LineMesh(points, lines, color, radius=0.02)
#         line_mesh_geoms = line_mesh.cylinder_segments
#         return line_mesh_geoms