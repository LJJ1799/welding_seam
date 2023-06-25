import os

from xml_parser import parse_frame_dump, list2array
from foundation import load_pcd_data, points2pcd, draw, fps
from math_util import rotate_mat, rotation_matrix_from_vectors
import pickle
from xml.dom.minidom import Document
import copy
import time
from shutil import copyfile
import numpy as np
import open3d as o3d
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

    def rotation(self,axis,norm):
        rot_axis = np.cross(axis, norm) / (np.linalg.norm(axis) * np.linalg.norm(norm))
        theta = np.arccos((axis @ norm)) / (np.linalg.norm(axis) * np.linalg.norm(norm))
        rotation = rotate_mat(axis=rot_axis, radian=theta)
        return rotation

    def crop(self, weld_info, crop_size=400, num_points=4096, vis=False):
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
        weld_seam_points=weld_info[:,1:4]
        weld_seam=o3d.geometry.PointCloud()

        weld_seam.points=o3d.utility.Vector3dVector(weld_seam_points)
        x_center = (np.max(weld_info[:,1]) + np.min(weld_info[:,1])) / 2
        y_center = (np.max(weld_info[:,2]) + np.min(weld_info[:,2])) / 2
        z_center = (np.max(weld_info[:,3]) + np.min(weld_info[:,3])) / 2
        translate=np.array([x_center,y_center,z_center])
        print('translate',translate)
        # print(weld_info[2][1:4])
        x_diff=np.max(weld_info[:,1])-np.min(weld_info[:,1])
        if x_diff<2:
            x_diff=0
        y_diff = np.max(weld_info[:, 2])-np.min(weld_info[:, 2])
        if y_diff<2:
            y_diff=0
        z_diff = np.max(weld_info[:, 3])-np.min(weld_info[:, 3])
        if z_diff<2:
            z_diff=0

        distance=pow(pow(x_diff,2)+pow(y_diff,2)+pow(z_diff,2),0.5)
        extent = 70
        # crop_extent = np.array([max(x_diff,extent), max(y_diff,extent),max(z_diff,extent)])
        crop_extent=np.array([400,400,400])
        # move the coordinate center to the welding spot
        pc.translate(-translate)
        weld_seam.translate(-translate)
        # rotation at this welding spot  1.
        rot = weld_info[0,10:13] * np.pi / 180
        rotation = rotate_mat(axis=[1, 0, 0], radian=rot[0])
        tf1 = np.zeros((4, 4))
        ## rotation at this welding spot  2.
        # axis=np.array([0,0,1])
        # print(weld_info[0,4:7])
        # norm_1=np.around(weld_info[0,4:7], decimals=6)
        # print('norm_1',norm_1)
        # rot_axis=np.cross(axis,norm_1)/(np.linalg.norm(axis)*np.linalg.norm(norm_1))
        # print('rot_axis',rot_axis)
        # theta=np.arccos((axis@norm_1))/(np.linalg.norm(axis)*np.linalg.norm(norm_1))
        # print('theta',theta)
        # rotation=rotate_mat(axis=rot_axis,radian=theta)
        # print('rotation',rotation)

        tf1[3, 3] = 1.0
        tf1[0:3, 0:3] = rotation
        # pc.transform(tf)
        # weld_seam.transform(tf)
        # new normals
        norm1 = np.around(weld_info[0, 4:7], decimals=6)
        print('norm1',norm1)
        norm2 = np.around(weld_info[0, 7:10], decimals=6)
        print('norm2',norm2)
        norm1_r = np.matmul(rotation, norm1.T)
        norm2_r = np.matmul(rotation, norm2.T)
        # torch pose
        pose = np.zeros((3, 3))
        for i in range(weld_info.shape[0]):
            pose[0:3, 0] = weld_info[i,14:17]
            pose[0:3, 1] = weld_info[i,17:20]
            pose[0:3, 2] = weld_info[i,20:23]
        # cauculate the new pose after rotation
            pose_new = np.matmul(rotation, pose)

            weld_info[i,4:7] = norm1_r
            weld_info[i,7:10] = norm2_r
            weld_info[i,14:17] = pose_new[0:3, 0]
            weld_info[i,17:20] = pose_new[0:3, 1]
            weld_info[i,20:23] = pose_new[0:3, 2]

        coor1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
        mesh_arrow1 = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=20 * 1,
            cone_radius=1.5 * 1,
            cylinder_height=20 * 1,
            cylinder_radius=1.5 * 1
        )
        mesh_arrow1.paint_uniform_color([0, 0, 1])

        mesh_arrow2 = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=20 * 1,
            cone_radius=1.5 * 1,
            cylinder_height=20 * 1,
            cylinder_radius=1.5 * 1
        )
        mesh_arrow2.paint_uniform_color([0, 1, 0])

        norm_ori = np.array([0, 0, 1])
        # bounding box of cutting area
        R = rotation_matrix_from_vectors(norm_ori, norm_ori)
        rotation_bbox1 = rotation_matrix_from_vectors(norm_ori, norm_ori)
        rotation_bbox2 = rotation_matrix_from_vectors(norm_ori, norm_ori)
        bbox = o3d.geometry.OrientedBoundingBox(center=np.array([0,0,0]), R=R,
                                                extent=crop_extent)
        crop_extent1=np.array([100,200,300])
        crop_extent2 = np.array([100,200,300])

        #the situation that the norm vector is parallel to axis
            #norm [0,0,1], plane in XOY
        if abs(norm1[2]) != 0 and abs(norm1[1]) == 0 and abs(norm1[0]) == 0:
            crop_extent1=np.array([max(x_diff,extent),max(y_diff,extent),1])
            print('1')
        if abs(norm2[2]) != 0 and abs(norm2[1]) == 0 and abs(norm2[0]) == 0:
            crop_extent2=np.array([max(x_diff,extent),max(y_diff,extent),1])
            print('2')
        #norm [0,1,0], plane in XOZ
        if abs(norm1[1]) != 0 and abs(norm1[0]) == 0 and abs(norm1[2]) == 0:
            crop_extent1=np.array([max(x_diff,extent),1,max(z_diff,extent)])
            print('3')
        if abs(norm2[1]) != 0 and abs(norm2[0]) == 0 and abs(norm2[2]) == 0:
            crop_extent2=np.array([max(x_diff,extent),1,max(z_diff,extent)])
            print('4')

        # if y_diff > 0 and x_diff == 0 and z_diff == 0:
            #norm [1,0,0], plane in YOZ
        if abs(norm1[0]) != 0 and abs(norm1[1]) == 0 and abs(norm1[2]) == 0:
            crop_extent1 = np.array([1,max(y_diff, extent), max(z_diff, extent)])
            print('5')
        if abs(norm2[0]) != 0 and abs(norm2[1]) == 0 and abs(norm2[2]) == 0:
            crop_extent2 = np.array([1,max(y_diff, extent), max(z_diff, extent)])
            print('6')
        # #norm [0,0,1], plane in XOY
        # if abs(norm1[2]) != 0 and abs(norm1[0]) == 0 and abs(norm1[1]) == 0:
        #     crop_extent1 = np.array([extent,max(distance, extent), 1])
        #     print('7')
        # if abs(norm2[2]) != 0 and abs(norm2[0]) == 0 and abs(norm2[1]) == 0:
        #     crop_extent2 = np.array([extent,max(distance, extent), 1])
        #     print('8')
        #
        # # if z_diff > 0 and x_diff == 0 and y_diff == 0:
        #     #norm [1,0,0], plane in YOZ
        # if abs(norm1[0]) != 0 and abs(norm1[1]) == 0 and abs(norm1[2]) == 0:
        #     crop_extent1 = np.array([1, extent,max(distance, extent)])
        #     print('9')
        # if abs(norm2[0]) != 0 and abs(norm2[1]) == 0 and abs(norm2[2]) == 0:
        #     crop_extent2 = np.array([1, extent,max(distance, extent)])
        #     print('10')
        # #norm [0,1,0], plan in XOZ
        # if abs(norm1[1]) != 0 and abs(norm1[0]) == 0 and abs(norm1[2]) == 0:
        #     crop_extent1 = np.array([extent, 1,max(distance, extent)])
        #     print('11')
        # if abs(norm2[1]) != 0 and abs(norm2[0]) == 0 and abs(norm2[2]) == 0:
        #     crop_extent2 = np.array([extent, 1,max(distance, extent)])
        #     print('12')

        axis_mesh = np.array([0, 0, 1])
        if abs(norm1[2]) == 0 and (abs(norm1[0]) != 0 or abs(norm1[1]) != 0):
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            print('aa')
        elif norm1[2] < 0 and abs(norm1[0]) == 0 and abs(norm1[1]) == 0:
            rotation_mesh1=np.array([1,0,0,0,1,0,0,0,-1]).reshape(3,3)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            print('bb')

        if abs(norm2[2]) == 0 and (abs(norm2[0]) != 0 or abs(norm2[1]) != 0):
            rotation_mesh2 = self.rotation(axis_mesh, norm2)
            mesh_arrow2.rotate(rotation_mesh2, center=np.array([0, 0, 0]))
            print('cc')
        elif norm2[2] < 0 and abs(norm2[0]) == 0 and abs(norm2[1]) == 0:
            rotation_mesh2=np.array([1,0,0,0,1,0,0,0,-1]).reshape(3,3)
            mesh_arrow2.rotate(rotation_mesh2, center=np.array([0, 0, 0]))
            print('dd')

        #the situation that the norm vector is not parallel to axis
        #norm on XOZ plane
        if abs(norm1[1]) == 0 and abs(norm1[0]) != 0 and abs(norm1[2]) != 0:
            axis_bbox1=np.array([1,0,0])
            crop_extent1=np.array([1,extent,distance])
            rotation_mesh1=self.rotation(axis_mesh,norm1)
            mesh_arrow1.rotate(rotation_mesh1,center=np.array([0, 0, 0]))
            rotation_bbox1=self.rotation(axis_bbox1,norm1)
        if abs(norm2[1]) == 0 and abs(norm2[0]) != 0 and abs(norm2[2]) != 0:
            axis_bbox2=np.array([1,0,0])
            crop_extent2=np.array([1,extent,distance])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)

        #norm on XOY plane
        if abs(norm1[2]) == 0 and abs(norm1[0]) != 0 and abs(norm1[1]) != 0:
            axis_bbox1 = np.array([1, 0, 0])
            crop_extent1 = np.array([1, distance, extent])
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            rotation_bbox1 = self.rotation(axis_bbox1, norm1)
        if abs(norm2[2]) == 0 and abs(norm2[0]) != 0 and abs(norm2[1]) != 0:
            axis_bbox2=np.array([1,0,0])
            crop_extent2=np.array([1,distance,extent])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)

        #norm on YOZ plane
        if abs(norm1[0]) == 0 and abs(norm1[1]) != 0 and abs(norm1[2]) != 0:
            axis_bbox1 = np.array([0, 0, 1])
            crop_extent1 = np.array([extent, distance, 1])
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            rotation_bbox1 = self.rotation(axis_bbox1, norm1)
        if abs(norm2[0]) == 0 and abs(norm2[1]) != 0 and abs(norm2[2]) != 0:
            axis_bbox2=np.array([0,0,1])
            crop_extent2=np.array([extent,distance,1])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)

        #norm [x,y,z]
        if abs(norm1[0]) != 0 and abs(norm1[1]) != 0 and abs(norm1[2]) != 0:
            axis_bbox1 = np.array([1, 0, 0])
            crop_extent1 = np.array([1, distance, 1])
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            rotation_bbox1 = self.rotation(axis_bbox1, norm1)
        if abs(norm2[0]) == 0 and abs(norm2[1]) != 0 and abs(norm2[2]) != 0:
            axis_bbox2=np.array([0,0,1])
            crop_extent2=np.array([extent,distance,1])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)


        #
        # print('norm1',norm1)
        # print(rotation1)
        # print('norm2',norm2)
        # print(rotation2)
        # tmp_axis=np.array([1,0,0])
        # tmp_rot=self.rotation(tmp_axis,norm1)
        bbox1 = o3d.geometry.OrientedBoundingBox(center=np.array([0,0,0]),R=rotation_bbox1,extent=crop_extent1)
        bbox2 = o3d.geometry.OrientedBoundingBox(center=np.array([0, 0, 0]), R=rotation_bbox2, extent=crop_extent2)

        pc.paint_uniform_color([1,0,0])
        weld_seam.paint_uniform_color([0,1,0])
        cropped_pc_large=pc.crop(bbox)
        cropped_pc1 = pc.crop(bbox1)
        cropped_pc2 = pc.crop(bbox2)
        cropped_pc = cropped_pc1+cropped_pc2
        idx_crop = bbox.get_point_indices_within_bounding_box(cropped_pc.points)
        xyz_crop = self.xyz[idx_crop]
        xyz_crop -= translate
        xyz_crop_new = np.matmul(rotation, xyz_crop.T).T
        l_crop = self.l[idx_crop]
        xyzl_crop = np.c_[xyz_crop_new, l_crop]
        xyzl_crop = np.unique(xyzl_crop, axis=0)
        while xyzl_crop.shape[0] < num_points:
            xyzl_crop = np.vstack((xyzl_crop, xyzl_crop))
        xyzl_crop = fps(xyzl_crop, num_points)
        if vis:
            o3d.visualization.draw_geometries([cropped_pc_large,bbox,cropped_pc,weld_seam,coor1,mesh_arrow1,mesh_arrow2,bbox1,bbox2])
        return xyzl_crop, cropped_pc, weld_info

def slice_one(pc_path, path_wz, path_lookup_table, xml_path, name, crop_size=400, num_points=2048):
    '''Slicing one component

    Args:
        pc_path (str): path to a pcd format point cloud
        xml_path (path): path to the xml file corresponding to the pc
        name (str): name of the pc
    '''
    # create welding scene
    ws = WeldScene(pc_path)
    # load welding info contains position, pose, normals, torch, etc.
    frames = list2array(parse_frame_dump(xml_path))
    # a summary of the filename of all the welding slices in one component with theirs welding infomation
    d = {}
    minpoints = 1000000
    for i in range(frames.shape[0]):
        weld_info = frames[i, 3:].astype(float)
        cxyzl, cpc, new_weld_info = ws.crop(weld_info=weld_info, crop_size=crop_size, num_points=num_points)
        # draw(cxyzl[:,0:3], cxyzl[:,3])
        # save the pc slice

        points2pcd(os.path.join(path_wz, name + '_' + str(i) + '.pcd'), cxyzl)
        d[name + '_' + str(i)] = new_weld_info
    with open(os.path.join(path_lookup_table, name + '.pkl'), 'wb') as tf:
        pickle.dump(d, tf, protocol=2)
    # print ('num of welding spots: ',len(d))
pc_file='Reisch.pcd'
ws=WeldScene(pc_file)
xml_path='Reisch.xml'
frames = list2array(parse_frame_dump(xml_path))
path_lookup_table='./lookup_table2'
path_wz='./welding_zone2'
if not os.path.exists(path_wz):
    os.mkdir(path_wz)
if not os.path.exists(path_lookup_table):
    os.mkdir(path_lookup_table)
weld_infos=[]
for i in range(0,200):
    tmp=frames[frames[:,-2]==str(i)]
    if len(tmp)!=0:
        weld_infos.append(tmp)
weld_infos=np.array(weld_infos)
d = {}
name='Reisch'
# weld_seam_points=weld_infos[0][:,4:7]
# print(weld_seam_points)
# print(weld_info)
# print(weld_info[31][:,3:])
for i in range(0,199):
    weld_info=weld_infos[i][:,3:].astype(float)
    cxyzl, cpc, new_weld_info = ws.crop(weld_info=weld_info, crop_size=400, num_points=4096)
    points2pcd(os.path.join(path_wz,name + '_' + str(i) + '.pcd'), cxyzl)
    d[name + '_' + str(i)] = new_weld_info
with open(os.path.join(path_lookup_table, name + '.pkl'), 'wb') as tf:
    pickle.dump(d, tf, protocol=2)


# for i in range (100,125):
#     weld_info = frames[i, 3:].astype(float)
#     print(weld_info[1:4])
#     cxyzl, cpc, new_weld_info = ws.crop(weld_info=weld_info, crop_size=400, num_points=2048)
# for i in range(frames.shape[0]):
#     weld_info = frames[i, 3:].astype(float)
#     print(weld_info)