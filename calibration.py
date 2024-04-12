import datetime
import time
import cv2
import numpy as np

import apriltag
import rscam
import mathutils

TARGET_POSE = np.array(
 [[-0.99923693,  0.01205297, -0.03715211,  0.12130056],
 [ 0.00212936 ,-0.93297031 ,-0.35994704 ,-0.0099027 ],
 [-0.03900024 ,-0.35975149 , 0.93223272 , 0.98273432],
 [ 0.         , 0.         , 0.         , 1.        ]]
        )
TRANSFORM_TO_ARM = None


def calc_measures(d):
    homogeneous = mathutils.to_homogeneous(d.pose_t, d.pose_R)
    xyz_pos = homogeneous[:3, 3].reshape(3)
    xyz_rot = mathutils.mat_to_roll_pitch_yaw(homogeneous)
    return xyz_pos, xyz_rot, homogeneous


def draw_axis(cam_intr, homogeneous, image):
    base_pose = np.linalg.inv(homogeneous) @ np.eye(4)  # Base tag.
    return mathutils.draw_axis(
        image,
        base_pose[:3, :3],
        base_pose[:3, 3],
        cam_intr,
        0.05,
        5,
        text_label=True,
        draw_arrow=True,
    )


def put_texts(image, homogeneous, target_h):
    pos_error_threshold = 0.003  # 3mm
    rot_error_threshold = 0.8  # 0.8 degree
    cv2.putText(
        image,
        "x pos: {0}{1:0.3f}".format(
            "+" if homogeneous[0, 3] - target_h[0, 3] > 0 else "",
            (homogeneous[0, 3] - target_h[0, 3]),
        ),
        org=(50, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        thickness=4,
        color=(
            (255, 0, 0)
            if abs(homogeneous[0, 3] - target_h[0, 3]) > pos_error_threshold else
            (0, 255, 0)
        ),
    )
    cv2.putText(
        image,
        "y pos: {0}{1:0.3f}".format(
            "+" if homogeneous[1, 3] - target_h[1, 3] > 0 else "",
            (homogeneous[1, 3] - target_h[1, 3]),
        ),
        org=(50, 100),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        thickness=4,
        color=(
            (255, 0, 0)
            if abs(homogeneous[1, 3] - target_h[1, 3]) > pos_error_threshold else
            (0, 255, 0)
        ),
    )

    cv2.putText(
        image,
        "z pos: {0}{1:0.3f}".format(
            "+" if homogeneous[2, 3] - target_h[2, 3] > 0 else "",
            (homogeneous[2, 3] - target_h[2, 3]),
        ),
        org=(50, 150),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        thickness=4,
        color=(
            (255, 0, 0)
            if abs(homogeneous[2, 3] - target_h[2, 3]) > pos_error_threshold else
            (0, 255, 0)
        ),
    )

    # calc rotation
    xyz_rot = mathutils.mat_to_roll_pitch_yaw(homogeneous)
    target_xyz_rot = mathutils.mat_to_roll_pitch_yaw(target_h)

    cv2.putText(
        image,
        "x rot: {0}{1:0.3f}".format(
            "+"
            if (np.degrees(xyz_rot[0]) - np.degrees(target_xyz_rot[0])) > 0
            else "",
            (np.degrees(xyz_rot[0]) - np.degrees(target_xyz_rot[0])),
        ),
        org=(50, 200),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        thickness=4,
        color=(
            (255, 0, 0)
            if abs(np.degrees(xyz_rot[0]) - np.degrees(target_xyz_rot[0])) > rot_error_threshold else
            (0, 255, 0)
        ),
    )
    cv2.putText(
        image,
        "y rot: {0}{1:0.3f}".format(
            "+"
            if (np.degrees(xyz_rot[1]) - np.degrees(target_xyz_rot[1])) > 0
            else "",
            (np.degrees(xyz_rot[1]) - np.degrees(target_xyz_rot[1])),
        ),
        org=(50, 250),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        thickness=4,
        color=(
            (255, 0, 0)
            if abs(np.degrees(xyz_rot[1]) - np.degrees(target_xyz_rot[1])) > rot_error_threshold else
            (0, 255, 0)
        ),
    )
    cv2.putText(
        image,
        "z rot: {0}{1:0.3f}".format(
            "+"
            if (np.degrees(xyz_rot[2]) - np.degrees(target_xyz_rot[2])) > 0
            else "",
            (np.degrees(xyz_rot[2]) - np.degrees(target_xyz_rot[2])),
        ),
        org=(50, 300),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        thickness=4,
        color=(
            (255, 0, 0)
            if abs(np.degrees(xyz_rot[2]) - np.degrees(target_xyz_rot[2])) > rot_error_threshold else
            (0, 255, 0)
        ),
    )
    # return image

def main():
    at = apriltag.AprilTag(0.053)  # large tag from `at.pdf`
    cam = rscam.Camera((1280, 720), (1280, 720), 30)
    time.sleep(2)
    print('camera init done.')

    while True:
        try:
            color_image, depth_image = cam.get_image()
            homogeneous = None
            result = at.detect(color_image, cam.intr_param)
            for tag in result:
                if tag.tag_id == 1:
                    pos, rot, homogeneous = calc_measures(tag)
                    print(f'pos: {pos}, rot: {rot} \n at {datetime.datetime.now()}')

            if homogeneous is not None:
                labelled = draw_axis(cam.intr_mat, homogeneous, color_image)
                # labelled = mathutils.put_texts(labelled, homogeneous, TARGET_POSE)
                put_texts(labelled, homogeneous, TARGET_POSE)
            else:
                labelled = color_image
            cv2.imshow(
                "calibration",
                cv2.cvtColor(cv2.resize(labelled, (1280, 720)), cv2.COLOR_BGR2RGB),
            )
            time.sleep(0.1)

            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
            elif k == ord('s') and homogeneous is not None:
                print(type(homogeneous), '\n', homogeneous)
                break
            continue

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()

# pos 三维： [0] x 轴： ，[1] y 轴 ，[2] z 轴上下
# rot 三维： [0] 绕  轴，[1] 绕  轴，[2] 绕  轴
# cam_intr_param_d435 = [924.11083984375, 922.5145263671875, 641.4503173828125, 351.6056213378906]
