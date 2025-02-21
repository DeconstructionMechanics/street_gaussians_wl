# This script is used for processing the [lidar_position] and [beams_direction] for training

import numpy as np

lidar_path = "ego_pose/"
beams_path = "beams.npz"

def ego_pose_reader(file_path):
    try:
        data = np.loadtxt(file_path, max_rows=3)
        lidar_pose = data[:, -1]
        #print(lidar_pose.tolist())
        return lidar_pose
    except FileNotFoundError:
        print(f"file {file_path} cannot found")
        return [0,0,0]
    except ValueError:
        print("Wrong file format")
        return [0,0,0]

def data_combine(lidar_pos, beam_pos):
    dir = beam_pos - lidar_pos
    magnitudes = np.linalg.norm(dir, axis=1, keepdims=True)
    magnitudes[magnitudes == 0] = 1
    dir = dir / magnitudes

    lidar_expanded = np.tile(lidar_pos, (len(dir), 1))
    lidar_dir_combined = np.hstack((lidar_expanded,dir))

    return lidar_dir_combined

# process lidar_position data
beams = np.load(beams_path,allow_pickle=True)

lidar_top = beams['top'].item()
lidar_front = beams['front'].item()
lidar_sideLeft = beams['side_left'].item()
lidar_sideRight = beams['side_right'].item()
lidar_rear = beams['rear'].item()

total_frames = 198

top_data_list = []
front_data_list = []
sideLeft_data_list = []
sideRight_data_list = []
rear_data_list = []

for frame in range(total_frames):
    frame_num = str(frame).zfill(6)

    top_pose = ego_pose_reader(f"{lidar_path}{frame_num}_0.txt")
    front_pose = ego_pose_reader(f"{lidar_path}{frame_num}_1.txt")
    sideLeft_pose = ego_pose_reader(f"{lidar_path}{frame_num}_2.txt")
    sideRight_pose = ego_pose_reader(f"{lidar_path}{frame_num}_3.txt")
    rear_pose = ego_pose_reader(f"{lidar_path}{frame_num}_4.txt")

    top_data_list.append(data_combine(top_pose, lidar_top[frame]))
    front_data_list.append(data_combine(front_pose, lidar_front[frame]))
    sideLeft_data_list.append(data_combine(sideLeft_pose, lidar_sideLeft[frame]))
    sideRight_data_list.append(data_combine(sideRight_pose, lidar_sideRight[frame]))
    rear_data_list.append(data_combine(rear_pose, lidar_rear[frame]))

top_data = np.concatenate(top_data_list, axis=0)
front_data = np.concatenate(front_data_list, axis=0)
sideLeft_data = np.concatenate(sideLeft_data_list, axis=0)
sideRight_data = np.concatenate(sideRight_data_list, axis=0)
rear_data = np.concatenate(rear_data_list, axis=0)

total_data = np.concatenate([top_data, front_data, sideLeft_data, sideRight_data, rear_data], axis=0)

print("top_data shape:", top_data.shape)
print("front_data shape:", front_data.shape)
print("sideLeft_data shape:", sideLeft_data.shape)
print("sideRight_data shape:", sideRight_data.shape)
print("rear_data shape:", rear_data.shape)
print("total_data shape:", total_data.shape)

np.savez_compressed('total_data.npz',total = total_data)
    
    








