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

    def get_distance_and_translate(self,weld_info):
        x_center = (np.max(weld_info[:,1]) + np.min(weld_info[:,1])) / 2
        y_center = (np.max(weld_info[:,2]) + np.min(weld_info[:,2])) / 2
        z_center = (np.max(weld_info[:,3]) + np.min(weld_info[:,3])) / 2
        translate=np.array([x_center,y_center,z_center])
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
        distance = int(pow(pow(x_diff,2)+pow(y_diff,2)+pow(z_diff,2),0.5))+25

        return distance,translate

    def bbox_(self,norm1,norm2,distance,extent,mesh_arrow1,mesh_arrow2):
        norm_ori = np.array([0, 0, 1])
        vector_seam=abs(np.cross(norm1,norm2))
        crop_extent1=np.array([100,200,300])
        crop_extent2 = np.array([100,200,300])
        rotation_bbox1 = rotation_matrix_from_vectors(norm_ori, norm_ori)
        rotation_bbox2 = rotation_matrix_from_vectors(norm_ori, norm_ori)
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
        #the situation that the norm vector is parallel to axis
            #norm [0,0,1], plane in XOY

        if abs(norm1[2]) != 0 and abs(norm1[1]) == 0 and abs(norm1[0]) == 0:
            if np.max(vector_seam)==vector_seam[0]:
                crop_extent1 = np.array([distance,extent,1])
            elif np.max(vector_seam)==vector_seam[1]:
                crop_extent1 = np.array([extent, distance, 1])
        if abs(norm2[2]) != 0 and abs(norm2[1]) == 0 and abs(norm2[0]) == 0:
            if np.max(vector_seam)==vector_seam[0]:
                crop_extent2 = np.array([distance,extent,1])
            elif np.max(vector_seam)==vector_seam[1]:
                crop_extent2 = np.array([extent, distance, 1])
        #norm [0,1,0], plane in XOZ
        if abs(norm1[1]) != 0 and abs(norm1[0]) == 0 and abs(norm1[2]) == 0:
            if np.max(vector_seam)==vector_seam[0]:
                crop_extent1 = np.array([distance,1,extent])
            elif np.max(vector_seam)==vector_seam[2]:
                crop_extent1 = np.array([extent, 1, distance])
        if abs(norm2[1]) != 0 and abs(norm2[0]) == 0 and abs(norm2[2]) == 0:
            if np.max(vector_seam)==vector_seam[0]:
                crop_extent2 = np.array([distance,1,extent])
            elif np.max(vector_seam)==vector_seam[2]:
                crop_extent2 = np.array([extent, 1, distance])

        # if y_diff > 0 and x_diff == 0 and z_diff == 0:
            #norm [1,0,0], plane in YOZ
        if abs(norm1[0]) != 0 and abs(norm1[1]) == 0 and abs(norm1[2]) == 0:
            if np.max(vector_seam)==vector_seam[1]:
                crop_extent1 = np.array([1,distance, extent])
            elif np.max(vector_seam)==vector_seam[2]:
                crop_extent1 = np.array([1, extent, distance])
        if abs(norm2[0]) != 0 and abs(norm2[1]) == 0 and abs(norm2[2]) == 0:
            if np.max(vector_seam)==vector_seam[1]:
                crop_extent2 = np.array([1,distance, extent])
            elif np.max(vector_seam)==vector_seam[2]:
                crop_extent2 = np.array([1, extent, distance])

        #the situation that the norm vector is not parallel to axis
        #norm on XOZ plane
        if abs(norm1[1]) == 0 and abs(norm1[0]) != 0 and abs(norm1[2]) != 0:
            axis_bbox1=np.array([1,0,0])
            crop_extent1=np.array([1,extent,distance])
            rotation_mesh1=self.rotation(axis_mesh,norm1)
            mesh_arrow1.rotate(rotation_mesh1,center=np.array([0, 0, 0]))
            rotation_bbox1=self.rotation(axis_bbox1,norm1)
            if abs(norm1[0])>=abs(norm1[2]):
                axis_bbox2 = np.array([1, 0, 0])
            elif abs(norm1[0])<abs(norm1[2]):
                axis_bbox2 = np.array([[0,0,1]])
            rotation_bbox2 = self.rotation(axis_bbox2, norm1)

        if abs(norm2[1]) == 0 and abs(norm2[0]) != 0 and abs(norm2[2]) != 0:
            axis_bbox2=np.array([1,0,0])
            crop_extent2=np.array([1,extent,distance])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)
            if abs(norm2[0])>=abs(norm2[2]):
                axis_bbox1 = np.array([1, 0, 0])
            elif abs(norm2[0])<abs(norm2[2]):
                axis_bbox1 = np.array([[0,0,1]])
            rotation_bbox1 = self.rotation(axis_bbox1, norm2)

        #norm on XOY plane
        if abs(norm1[2]) == 0 and abs(norm1[0]) != 0 and abs(norm1[1]) != 0:
            axis_bbox1 = np.array([1, 0, 0])
            crop_extent1 = np.array([1, distance, extent])
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            rotation_bbox1 = self.rotation(axis_bbox1, norm1)
            if abs(norm1[0])>=abs(norm1[1]):
                axis_bbox2 = np.array([1,0,0])
            elif abs(norm1[0])<abs(norm1[1]):
                axis_bbox2 = np.array([0, 1, 0])
            rotation_bbox2 = self.rotation(axis_bbox2, norm1)

        if abs(norm2[2]) == 0 and abs(norm2[0]) != 0 and abs(norm2[1]) != 0:
            axis_bbox2=np.array([1,0,0])
            axis_bbox1=np.array([1,0,0])
            crop_extent2=np.array([1,distance,extent])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)
            if abs(norm2[0]) >= abs(norm2[1]):
                axis_bbox1 = np.array([1, 0, 0])
            elif abs(norm2[0]) < abs(norm2[1]):
                axis_bbox1 = np.array([0, 1, 0])
            rotation_bbox1 = self.rotation(axis_bbox1, norm2)

        #norm on YOZ plane
        if abs(norm1[0]) == 0 and abs(norm1[1]) != 0 and abs(norm1[2]) != 0:
            axis_bbox1 = np.array([0, 0, 1])
            crop_extent1 = np.array([extent, distance, 1])
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            rotation_bbox1 = self.rotation(axis_bbox1, norm1)
            if abs(norm1[1])>=abs(norm1[2]):
                axis_bbox2 = np.array([0, 1, 0])
            elif abs(norm1[1])<abs(norm1[2]):
                axis_bbox2 = np.array([0, 0, 1])
            rotation_bbox2 = self.rotation(axis_bbox2, norm1)

        if abs(norm2[0]) == 0 and abs(norm2[1]) != 0 and abs(norm2[2]) != 0:
            axis_bbox2=np.array([0,0,1])
            crop_extent2=np.array([extent,distance,1])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)
            if abs(norm2[1])>=abs(norm2[2]):
                axis_bbox1 = np.array([0, 1, 0])
            elif abs(norm2[1])<abs(norm2[2]):
                axis_bbox1 = np.array([0, 0, 1])
            rotation_bbox1 = self.rotation(axis_bbox1, norm2)

        return rotation_bbox1,rotation_bbox2,crop_extent1,crop_extent2

    def crop(self, weld_info, seams, num_points=2048, vis=False):
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
        distance,translate=self.get_distance_and_translate(weld_info)
        extent = 90
        # crop_extent = np.array([max(x_diff,extent), max(y_diff,extent),max(z_diff,extent)])
        crop_extent=np.array([distance,extent+5,extent+5])
        # move the coordinate center to the welding spot
        pc.translate(-translate)
        weld_seam.translate(-translate)
        # rotation at this welding spot  1.
        rot = weld_info[0,10:13] * np.pi / 180
        rotation = rotate_mat(axis=[1, 0, 0], radian=rot[0])

        tf1 = np.zeros((4, 4))
        tf1[3, 3] = 1.0
        tf1[0:3, 0:3] = rotation
        # pc.transform(tf)
        # weld_seam.transform(tf)
        # new normals
        norm1 = np.around(weld_info[0, 4:7], decimals=6)
        norm2 = np.around(weld_info[0, 7:10], decimals=6)


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
        crop_extent_big=np.array([500,500,500])
        big_box=o3d.geometry.OrientedBoundingBox(center=np.array([0,0,0]),R=rotation_matrix_from_vectors(norm_ori, norm_ori),extent=crop_extent_big)
        # bounding box of cutting area
        rotation_bbox = rotation_matrix_from_vectors(norm_ori, norm_ori)
        seams_direction=np.cross(norm1,norm2)
        if (abs(seams_direction[0])==0 and (abs(seams_direction[1])!=0 or abs(seams_direction[2])!=0)) or (abs(seams_direction[0])!=0 and (abs(seams_direction[1])!=0 or abs(seams_direction[2])!=0)):
            rotation_bbox=self.rotation(np.array([1,0,0]),seams_direction)
        center_bbox=norm2/np.linalg.norm(norm2)*extent/2+norm1/np.linalg.norm(norm1)*extent/2
        bbox = o3d.geometry.OrientedBoundingBox(center=center_bbox, R=rotation_bbox,extent=crop_extent)


        rotation_bbox1, rotation_bbox2, crop_extent1, crop_extent2=self.bbox_(norm1,norm2,distance,extent,mesh_arrow1
                                                                              ,mesh_arrow2)
        bbox1 = o3d.geometry.OrientedBoundingBox(center=norm2/np.linalg.norm(norm2)*extent/2,R=rotation_bbox1,extent=crop_extent1)
        bbox2 = o3d.geometry.OrientedBoundingBox(center=norm1/np.linalg.norm(norm1)*extent/2, R=rotation_bbox2, extent=crop_extent2)
        bbox_points=np.asarray(bbox.get_box_points())
        bbox1_points=np.asarray(bbox1.get_box_points())
        bbox2_points=np.asarray(bbox2.get_box_points())
        in_bbox_list = []
        in_bbox1_list=[]
        in_bbox2_list=[]
        for sub_seams in seams:
            for origin_spot in sub_seams:
                spot=copy.copy(origin_spot)
                spot-=translate
                # if np.min(bbox1_points[:,0])<spot[0]<np.max(bbox1_points[:,0]) and np.min(bbox1_points[:,1])<spot[1]<np.max(bbox1_points[:,1]) and np.min(bbox1_points[:,2])<spot[2]<np.max(bbox1_points[:,2]):
                #     in_bbox1_list.append(sub_seams)
                #     break
                # if np.min(bbox2_points[:,0])<spot[0]<np.max(bbox2_points[:,0]) and np.min(bbox2_points[:,1])<spot[1]<np.max(bbox2_points[:,1]) and np.min(bbox2_points[:,2])<spot[2]<np.max(bbox2_points[:,2]):
                #     in_bbox2_list.append(sub_seams)
                #     break
                if np.min(bbox_points[:, 0]) < spot[0] < np.max(bbox_points[:, 0]) and np.min(bbox_points[:, 1]) < spot[1] < np.max(bbox_points[:, 1]) and np.min(bbox_points[:, 2]) < spot[2] < np.max(bbox_points[:, 2]):
                    in_bbox_list.append(spot)
                #     break

        # if len(in_bbox1_list)!=0:
        #     in_bbox1_points=o3d.geometry.PointCloud()
        #     in_bbox1_points.points=o3d.utility.Vector3dVector(np.asarray(in_bbox1_list))
        #     in_bbox1_points.paint_uniform_color([0,0,1])
        #     polygon1 = np.vstack((weld_seam_points,np.asarray(in_bbox1_list)) )
        # else:
        #     in_bbox1_points=o3d.geometry.PointCloud()
        #     in_bbox1_points.points=o3d.utility.Vector3dVector(np.asarray(weld_seam_points-translate))
        #     in_bbox1_points.paint_uniform_color([0,0,1])
        # # # #
        # if len(in_bbox2_list)!=0:
        #     in_bbox2_points=o3d.geometry.PointCloud()
        #     in_bbox2_points.points=o3d.utility.Vector3dVector(np.asarray(in_bbox2_list))
        #     in_bbox2_points.paint_uniform_color([0,0,1])
        #     polygon2 = np.vstack((weld_seam_points, np.asarray(in_bbox2_list)))
        # else:
        #     in_bbox2_points=o3d.geometry.PointCloud()
        #     in_bbox2_points.points=o3d.utility.Vector3dVector(weld_seam_points-translate)
        #     in_bbox2_points.paint_uniform_color([0,0,1])

        if len(in_bbox_list)!=0:
            in_bbox_points=o3d.geometry.PointCloud()
            in_bbox_points.points=o3d.utility.Vector3dVector(np.asarray(in_bbox_list))
            in_bbox_points.paint_uniform_color([0,0,1])
        else:
            in_bbox_points=o3d.geometry.PointCloud()
            in_bbox_points.points=o3d.utility.Vector3dVector(weld_seam_points-translate)
            in_bbox_points.paint_uniform_color([0,0,1])

        pc.paint_uniform_color([1,0,0])
        weld_seam.paint_uniform_color([0,1,0])
        croped_pc_big=pc.crop(big_box)
        cropped_pc_large=pc.crop(bbox)
        cropped_pc1 = pc.crop(bbox1)
        cropped_pc2 = pc.crop(bbox2)
        cropped_pc = cropped_pc1+cropped_pc2
        idx_crop_large=bbox.get_point_indices_within_bounding_box(pc.points)
        idx_crop1=bbox1.get_point_indices_within_bounding_box(pc.points)
        idx_crop2=bbox2.get_point_indices_within_bounding_box(pc.points)
        idx_crop = idx_crop1+idx_crop2
        xyz_crop = self.xyz[idx_crop_large]
        xyz_crop -= translate
        xyz_crop_new = np.matmul(rotation_matrix_from_vectors(norm_ori, norm_ori), xyz_crop.T).T
        l_crop = self.l[idx_crop_large]

        xyzl_crop = np.c_[xyz_crop_new, l_crop]
        xyzl_crop = np.unique(xyzl_crop, axis=0)

        while xyzl_crop.shape[0] < num_points:
            xyzl_crop = np.vstack((xyzl_crop, xyzl_crop))
        xyzl_crop = fps(xyzl_crop, num_points)
        if vis:
            o3d.visualization.draw_geometries([croped_pc_big,big_box,cropped_pc_large,cropped_pc,coor1,weld_seam,mesh_arrow1,mesh_arrow2,bbox,bbox1,bbox2])
        return xyzl_crop, cropped_pc, weld_info

pc_file='Reisch.pcd'
ws=WeldScene(pc_file)
xml_path='Reisch.xml'
frames = list2array(parse_frame_dump(xml_path))
path_lookup_table='./lookup_table'
path_wz='./welding_zone'
if not os.path.exists(path_wz):
    os.mkdir(path_wz)
if not os.path.exists(path_lookup_table):
    os.mkdir(path_lookup_table)
weld_infos=[]
weld_seams=[]
for i in range(len(frames)):
    tmp=frames[frames[:,-2]==str(i)]
    if len(tmp)!=0:
        weld_seams.append(tmp[:,4:7].astype(float))
        weld_infos.append(tmp)
d = {}

name='Reisch'
for i in range(0,11):
    weld_info=weld_infos[i][:,3:].astype(float)
    slice_name=weld_infos[i][0,0]
    # print(weld_infos[i][0,0])
    norm1 = np.around(weld_info[0, 4:7], decimals=6)
    norm2 = np.around(weld_info[0, 7:10], decimals=6)
    # print(norm1,norm2)
    # if (abs(norm1[1]) == 0 and abs(norm1[0]) != 0 and abs(norm1[2]) != 0) or (abs(norm2[1]) == 0 and abs(norm2[0]) != 0 and abs(norm2[2]) != 0):
    seams=np.delete(weld_seams,i,0)
    cxyzl, cpc, new_weld_info = ws.crop(weld_info=weld_info, seams=seams, num_points=2048)
    points2pcd(os.path.join(path_wz,slice_name + '.pcd'), cxyzl)
    d[slice_name + '_' + str(i)] = new_weld_info
with open(os.path.join(path_lookup_table, name + '.pkl'), 'wb') as tf:
    pickle.dump(d, tf, protocol=2)