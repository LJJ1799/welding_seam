from xml_parser import parse_frame_dump, list2array
from foundation import load_pcd_data, points2pcd, draw, fps
from math_util import rotate_mat, rotation_matrix_from_vectors
import pickle
from xml.dom.minidom import Document
import copy
import time
from shutil import copyfile
import open3d as o3d
import numpy as np
import pandas as pd
class WeldScene:
    '''
    Component point cloud processing, mainly for slicing

    Attributes:
        path_pc: path to labeled pc

    '''

    def __init__(self, pc_path):
        self.pc = o3d.geometry.PointCloud()
        xyzl = load_pcd_data(pc_path)
        # print (xyzl.shape)
        self.xyz = xyzl[:, 0:3]
        self.l = xyzl[:, 3]
        self.pc.points = o3d.utility.Vector3dVector(self.xyz)

    def crop(self, weld_info, weld_info2,crop_size=400, num_points=2048, vis=True):
        '''Cut around welding spot

        Args:
            weld_info (np.ndarray): welding info, including torch type, weld position, surface normals, torch pose
            crop_size (int): side length of cutting bbox in mm
            num_points (int): the default point cloud contains a minimum of 2048 points, if not enough then copy and fill
            vis (Boolean): True for visualization of the slice while slicing
        Returns:
            xyzl_crop (np.ndarray): cropped pc with shape num_points*4, cols are x,y,z,label
            cropped_pc (o3d.geometry.PointCloud): cropped pc for visualization
            weld_info (np.ndarray): update the rotated component pose for torch (if there is)
        '''
        pc = copy.copy(self.pc)

        # tow surface normals at the welding spot
        norm1 = np.around(weld_info[4:7], decimals=6)
        norm2 = np.around(weld_info[7:10], decimals=6)
        extent = crop_size - 10
        crop_extent = np.array([extent, extent, extent])
        weld_spot = weld_info[1:4]
        # move the coordinate center to the welding spot
        pc.translate(-weld_spot)
        # rotation at this welding spot
        rot = weld_info[10:13] * np.pi / 180
        rotation = rotate_mat(axis=[1, 0, 0], radian=rot[0])
        # torch pose
        pose = np.zeros((3, 3))
        pose[0:3, 0] = weld_info[14:17]
        pose[0:3, 1] = weld_info[17:20]
        pose[0:3, 2] = weld_info[20:23]
        # cauculate the new pose after rotation
        pose_new = np.matmul(rotation, pose)
        tf = np.zeros((4, 4))
        tf[3, 3] = 1.0
        tf[0:3, 0:3] = rotation
        pc.transform(tf)
        # new normals
        norm1_r = np.matmul(rotation, norm1.T)
        norm2_r = np.matmul(rotation, norm2.T)

        weld_info[4:7] = norm1_r
        weld_info[7:10] = norm2_r
        weld_info[14:17] = pose_new[0:3, 0]
        weld_info[17:20] = pose_new[0:3, 1]
        weld_info[20:23] = pose_new[0:3, 2]

        coor1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])
        norm_ori = np.array([0, 0, 1])
        # bounding box of cutting area
        R = rotation_matrix_from_vectors(norm_ori, norm_ori)
        bbox = o3d.geometry.OrientedBoundingBox(center=(extent / 2 - 60) * norm1_r + (extent / 2 - 60) * norm2_r, R=R,
                                                extent=crop_extent)
        cropped_pc = pc.crop(bbox)
        idx_crop = bbox.get_point_indices_within_bounding_box(pc.points)
        xyz_crop = self.xyz[idx_crop]
        xyz_crop -= weld_spot
        xyz_crop_new = np.matmul(rotation, xyz_crop.T).T
        l_crop = self.l[idx_crop]
        xyzl_crop = np.c_[xyz_crop_new, l_crop]
        xyzl_crop = np.unique(xyzl_crop, axis=0)
        while xyzl_crop.shape[0] < num_points:
            xyzl_crop = np.vstack((xyzl_crop, xyzl_crop))
        xyzl_crop = fps(xyzl_crop, num_points)
        if vis:
            o3d.visualization.draw_geometries([cropped_pc, coor1, bbox])
        return xyzl_crop, cropped_pc, weld_info

pc_path='Reisch.pcd'
ws = WeldScene(pc_path)
xml_path='Reisch.xml'
frames = list2array(parse_frame_dump(xml_path))
weld_info=[]
for i in range(0,200):
    tmp=frames[frames[:,-2]==str(i)]
    if len(tmp)!=0:
        weld_info.append(tmp)
weld_info=np.array(weld_info)
weld_info=weld_info[:,3:].astype(float)
weld_info1=weld_info[0]
weld_info2=weld_info[2]
cxyzl, cpc, new_weld_info = ws.crop(weld_info=weld_info1,weld_info2=weld_info2, crop_size=crop_size, num_points=num_points)
# for i in range(weld_info.shape[0]):
#     weld_info=weld_info[0]
# print(weld_info)
# print(weld_info)
# print(weld_info.shape)
# print(len(weld_info))

# point_cloud = o3d.io.read_point_cloud("Reisch.pcd")
# , [-1401.280148093301, 83.93862361422825, 2182.061376385556]
# plane_point = np.array([[-796.2801480933575, 83.9386236142102, 2182.061376385546]])
# plane_normal = np.array([-1.010719286043127e-13,-1.000000000000427,-1.804112415015879e-14])
# sliced_cloud = point_cloud.crop(plane_point, plane_normal, negative=False)
# o3d.visualization.draw_geometries([point_cloud])