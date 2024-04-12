import datetime
import json
import random
import time
import cv2
import numpy as np

import apriltag
from robot_method import cartesian_action_movement, forward_kinematics
import rscam
import mathutils

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

import robot_connect
args = robot_connect.parseConnectionArguments()



def calc_measures(d):
    homogeneous = mathutils.to_homogeneous(d.pose_t, d.pose_R)
    xyz_pos = homogeneous[:3, 3].reshape(3)
    xyz_rot = mathutils.mat_to_roll_pitch_yaw(homogeneous)
    return xyz_pos, xyz_rot, homogeneous

x_max = 0.8
x_min = 0.5
x_unit = 0.05
y_max = 0.3
y_min = -0.5
y_unit = 0.05
z_max = 0.4
z_min = 0.08
z_unit = 0.02
theta_x_min = 65.0
theta_x_max = 140.0
theta_x_unit = 10.0
theta_y_min = -180.0
theta_y_max = 180.0
theta_y_unit = 10.0
theta_z_min = 30.0
theta_z_max = 150.0
theta_z_unit = 10.0

change_unit = [x_unit, y_unit, z_unit, theta_x_unit, theta_y_unit, theta_z_unit]
change_limit = [x_max, y_max, z_max, theta_x_max, theta_y_max, theta_z_max]
change_bound = [x_min, y_min, z_min, theta_x_min, theta_y_min, theta_z_min]

def main():
    at = apriltag.AprilTag(0.053)  # large tag from `at.pdf`
    cam = rscam.Camera((1280, 720), (1280, 720), 30)
    time.sleep(2)
    print('camera init done.')

    R_target2cam_list = []
    t_target2cam_list = []
    R_base2gripper_list = []
    t_base2gripper_list = []

    # with robot_connect.DeviceConnection.createTcpConnection(args) as router:
    #     base = BaseClient(router)

    #     for i in range(100):
    #         if i % 30 == 0:
    #             input(f'current step: {i}, point number: {len(R_target2cam_list)}, waiting For hand-craft to set robot position: ')
    #             del base
    #             base = BaseClient(router)
    #             time.sleep(3)
                                
    #         chosen_index = random.randint(0, 5)
    #         plus_minus =  1 if random.randint(0, 1) == 0 else -1
    #         dof_6 = list(forward_kinematics(base))
    #         dof_6[chosen_index] += change_unit[chosen_index] * plus_minus
            
    #         if  not (change_bound[chosen_index] <= dof_6[chosen_index] <= change_limit[chosen_index]):
    #             continue
            
    #         cartesian_action_movement(base, dof_6)
            
    #         color_image, depth_image = cam.get_image()
    #         homogeneous = None
    #         result = at.detect(color_image, cam.intr_param)
    #         for tag in result:
    #             if tag.tag_id == 1:
    #                 pos, rot, homogeneous = calc_measures(tag)
    #                 # print(f'pos: {pos}, rot: {rot} \n at {datetime.datetime.now()}')
            
    #         if homogeneous is None:
    #             continue
                
    #         dof_6 = forward_kinematics(base)
    #         robot_pos = dof_6[:3]
    #         robot_rot = mathutils.eular_angle_to_rotation_matrix(dof_6[3:])

    #         R_target2cam_list.append(homogeneous[:3, :3])
    #         t_target2cam_list.append(homogeneous[:3, 3])
    #         R_base2gripper_list.append(robot_rot.T)
    #         t_base2gripper_list.append(-robot_rot.T @ robot_pos)
    
    # 使用with语句打开文件，自动处理文件关闭
    with open("result/2024-04-12-08-43-51.json") as f:
        # 加载JSON数据
        data = json.load(f)
        
        # 遍历数据
        for d in data:
            R_target2cam_list.append(np.array(d['R_target2cam']))
            t_target2cam_list.append(np.array(d['t_target2cam']))
            R_base2gripper_list.append(np.array(d['R_base2gripper']))
            t_base2gripper_list.append(np.array([-1 * num  for num in d['t_base2gripper']]))
    
    # combined_list = zip(R_target2cam_list, t_target2cam_list, R_base2gripper_list, t_base2gripper_list)
    # dict_list = [{"R_target2cam": R.tolist(), "t_target2cam": t.tolist(), "R_base2gripper": Rb.tolist(), "t_base2gripper": tb.tolist()} for R, t, Rb, tb in combined_list]
    
    # filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # with open(f'result/{filename}.json', 'w', encoding="UTF-8") as f:
    #     json.dump(dict_list, f, ensure_ascii=False, indent=4)
    
    R, T = cv2.calibrateHandEye(R_base2gripper_list, t_base2gripper_list, R_target2cam_list, t_target2cam_list)

    # with open(f'result/{filename}.txt', 'w', encoding="UTF-8") as f:
    #     f.write(f"Point number: {len(R_target2cam_list)}\n")
    #     f.write(f'R: {R}\n')
    #     f.write(f'T: {T}\n')
    
    print('Point number:', len(R_target2cam_list))
    print('R:', R)
    print('T:', T)
    
    
    
if __name__ == '__main__':
    main()

# pos 三维： [0] x 轴： ，[1] y 轴 ，[2] z 轴上下
# rot 三维： [0] 绕  轴，[1] 绕  轴，[2] 绕  轴
# cam_intr_param_d435 = [924.11083984375, 922.5145263671875, 641.4503173828125, 351.6056213378906]
