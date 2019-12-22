# --------------------------------------------------------
# Preprocess the Lyft Raw Data into corresponding numpy matrix format
# Licensed under The MIT License [see LICENSE for details]
# Author: Jiayi Luo, Zhoutong Jiang
# --------------------------------------------------------

# Install lyftdataset note:
# hash -d pip3
# hash -r pip3
# python3.6 -m pip install black
# python3.6 -m pip install lyft-dataset-sdk

from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
import pandas as pd
import pdb
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

# Define the dataset root
DATA_PATH = './3d-object-detection-for-autonomous-vehicles/'
lyftdata = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH+'train_data')

def get_lidar_points(lidar_token):
    '''Get lidar point cloud in the frame of the ego vehicle'''
    sd_record = lyftdata.get("sample_data", lidar_token)
    sensor_modality = sd_record["sensor_modality"]
    
    # Get aggregated point cloud in lidar frame.
    sample_rec = lyftdata.get("sample", sd_record["sample_token"])
    chan = sd_record["channel"]
    ref_chan = "LIDAR_TOP"
    pc, times = LidarPointCloud.from_file_multisweep(
        lyftdata, sample_rec, chan, ref_chan, num_sweeps=1
    )
    # Compute transformation matrices for lidar point cloud
    cs_record = lyftdata.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = lyftdata.get("ego_pose", sd_record["ego_pose_token"])
    vehicle_from_sensor = np.eye(4)
    vehicle_from_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
    vehicle_from_sensor[:3, 3] = cs_record["translation"]
    
    # Apply rotation and yaw geometric information to change the coordinates back to rectangular coordinate system
    ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
    rot_vehicle_flat_from_vehicle = np.dot(
        Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
        Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
    )
    vehicle_flat_from_vehicle = np.eye(4)
    vehicle_flat_from_vehicle[:3, :3] = rot_vehicle_flat_from_vehicle
    points = view_points(
        pc.points[:3, :], np.dot(vehicle_flat_from_vehicle, vehicle_from_sensor), normalize=False
    )
    return points

def get_data(lidar_token):
    _, boxes, _ = lyftdata.get_sample_data(
        lidar_token, flat_vehicle_coordinates=True
    )
    lidar_points = get_lidar_points(lidar_token)
    boxList = []
    classList = []
    for box in boxes:
        center = np.array(box.center)
        wlh = np.array(box.wlh)
        boxCor = np.concatenate((center,wlh),axis = 0)
        boxList.append(boxCor)
        classList.append(box.name)
    return lidar_points, boxList, classList

# Define the corresponding classes
map_ = {'car':0,'pedestrian':1,'animal':2,'other_vehicle':3,'bus':4,'motorcycle':5,'truck':6,'emergency_vehicle':7,'bicycle':8,}
def add_label(point_cloud, boxList, classList, background = False):
    n = point_cloud.shape[1]
    input_label = np.ones([n, 1]) * -2
    input_object = np.ones([n, 1]) * -2

    # total number of objects belong to at least one of the class
    numObj = 0

    # Loop over all the points
    for i in range(n):

        # For each point check wheter it locates inside any box or not
        point_x, point_y, point_z = point_cloud[0, i], point_cloud[1, i], point_cloud[2, i]
        for boxID in range(len(boxList)):
            box = boxList[boxID]
            x, y, z, l, w, h = box[:]
            xmin, xmax, ymin, ymax, zmin, zmax = x - l/2, x + l/2, y - w/2, y + w/2, z - h/2, z + h/2
            # if in box
            idx = 0
            if point_x >= xmin and point_x <= xmax and point_y >= ymin and point_y <= ymax and point_z >= zmin and point_z <= zmax:
                input_label[i] = map_[classList[boxID]]
                input_object[i] = boxID
                numObj += 1
                idx = 1
                break
        # if not in box, and backgound = True, remove the background information
        if not idx and background == True:
            numObj += 1
            input_label[i] = -1
            input_object[i] = -1
    # create input data in format 4xn
    input_data = np.zeros([5, numObj])
    count = 0
    for i in range(n):
        if input_label[i] != -2 and input_object[i] != -2:
            input_data[0, count], input_data[1, count], input_data[2, count], input_data[3, count], input_data[4, count] =\
            point_cloud[0, i], point_cloud[1, i], point_cloud[2, i], input_label[i], input_object[i]
            count += 1
    return input_data

# Read the metadata from the dataset
train = pd.read_csv('./3d-object-detection-for-autonomous-vehicles/train.csv')

# Save the data to predefined path
count = 0
for i in range(1):
    if i % 5 == 0 or i == train.shape[0]:
        print("Starting %dth sample... at %.4f%%" % (i, i/train.shape[0]))

    token = train.iloc[i]['Id']
    my_sample = lyftdata.get('sample', token)

    # Three different LiDAR datasets for one scene
    if 'LIDAR_TOP' in my_sample['data']:
        temp = lyftdata.get('sample_data', my_sample['data']['LIDAR_TOP'])
        lidar_token = my_sample['data']['LIDAR_TOP']
        a,b,c = get_data(lidar_token)
        print(b)
        d = add_label(a,b,c)
        path = './traindata/train_' + str(count)
        np.save(path, d)
        count += 1
    if 'LIDAR_FRONT_LEFT' in my_sample['data']:
        temp = lyftdata.get('sample_data', my_sample['data']['LIDAR_FRONT_LEFT'])
        lidar_token = my_sample['data']['LIDAR_FRONT_LEFT']
        a,b,c = get_data(lidar_token)
        d = add_label(a,b,c)
        path = './traindata/train_' + str(count)
        np.save(path, d)
        count += 1
    if 'LIDAR_FRONT_RIGHT' in my_sample['data']:
        temp = lyftdata.get('sample_data', my_sample['data']['LIDAR_FRONT_RIGHT'])
        lidar_token = my_sample['data']['LIDAR_FRONT_RIGHT']
        a,b,c = get_data(lidar_token)
        d = add_label(a,b,c)
        path = './traindata/train_' + str(count)
        np.save(path, d)
        count += 1
