from typing import Union
import cv2
import numpy as np
import numpy.typing as npt
PI = np.pi


def to_homogeneous(
        pos: Union[list, npt.NDArray[np.float32]], rot: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Given position and rotation matrix, convert it into homogeneous matrix."""

    if isinstance(pos, list):
        pos = np.array(pos)
    transform = np.zeros((4, 4))
    if pos.ndim == 2:
        transform[:3, 3:] = pos
    else:
        assert pos.ndim == 1
        transform[:3, 3] = pos
    transform[:3, :3] = rot
    transform[3, 3] = 1

    return transform


def mat_to_roll_pitch_yaw(rot_mat):
    """Convert rotation matrix to roll-pitch-yaw angles.
    Args:
        rot_mat: 3x3 rotation matrix.
    Returns:
        roll, pitch, yaw: roll-pitch-yaw angles.
    """
    roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
    pitch = np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[2, 1] ** 2 + rot_mat[2, 2] ** 2))
    yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    return roll, pitch, yaw

def rotation_matrix_to_eular_angler(R):
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    x = y = z = 0
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        Warning('not singular')
    return x, y, z

def eular_angle_to_rotation_matrix(eular):
    # Z-Y-X欧拉角 先绕Z转alpha,再绕动的Y转beta,再绕动的X转gamma
    gamma, beta, alpha = eular[0] * PI / 180, eular[1] * PI / 180, eular[2] * PI / 180

    R = np.zeros((3,3))

    R[0][0] = np.cos(alpha) * np.cos(beta)
    R[0][1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    R[0][2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    R[1][0] = np.sin(alpha) * np.cos(beta)
    R[1][1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    R[1][2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
    R[2][0] = -np.sin(beta)
    R[2][1] = np.cos(beta) * np.sin(gamma)
    R[2][2] = np.cos(beta) * np.cos(gamma)
    
    return R

def draw_axis(
        img: npt.NDArray[np.uint8],
        R: npt.NDArray[np.float32],
        t: npt.NDArray[np.float32],
        K: npt.NDArray[np.float32],
        s: float = 0.015,
        d: int = 3,
        rgb=True,
        axis="xyz",
        colors=None,
        trans=False,
        text_label: bool = False,
        draw_arrow: bool = False,
) -> npt.NDArray[np.uint8]:
    """Draw x, y, z axis on the image.

    Args:
        img: Image to draw on.
        R: Rotation matrix.
        t: Translation vector.
        K: Intrinsic matrix.
        s: Length of the axis.
        d: Thickness of the axis.


    Returns:
        Image with the axis drawn.
    """
    draw_img = img.copy()
    # Unit is meter
    rotV, _ = cv2.Rodrigues(R)
    # The tag's coordinate frame is centered at the center of the tag,
    # with x-axis to the right, y-axis down, and z-axis into the tag.
    if isinstance(s, float):
        points = np.float32([[s, 0, 0], [0, s, 0], [0, 0, s], [0, 0, 0]]).reshape(-1, 3)
    elif isinstance(s, list):
        # list
        points = np.float32(
            [[s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]], [0, 0, 0]]
        ).reshape(-1, 3)
    else:
        raise ValueError

    axis_points, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    a0 = np.array((int(axis_points[0][0][0]), int(axis_points[0][0][1])))
    a1 = np.array((int(axis_points[1][0][0]), int(axis_points[1][0][1])))
    a2 = np.array((int(axis_points[2][0][0]), int(axis_points[2][0][1])))
    a3 = np.array((int(axis_points[3][0][0]), int(axis_points[3][0][1])))
    if colors is None:
        if rgb:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        else:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    else:
        colors = [colors] * 3

    axes_map = {"x": (a0, colors[0]), "z": (a2, colors[2]), "y": (a1, colors[1])}

    for axis_label, (point, color) in axes_map.items():
        if axis_label in axis:
            if draw_arrow:
                draw_img = cv2.arrowedLine(
                    draw_img, tuple(a3), tuple(point), color, d, tipLength=0.5
                )
            else:
                draw_img = cv2.line(draw_img, tuple(a3), tuple(point), color, d)

    # Add labels for each axis
    if text_label:
        cv2.putText(
            draw_img, "X", tuple(a0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[0], 3
        )
        cv2.putText(
            draw_img, "Y", tuple(a2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[1], 3
        )
        cv2.putText(
            draw_img, "Z", tuple(a1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[2], 3
        )

    if trans:
        # Transparency value
        alpha = 0.50
        # Perform weighted addition of the input image and the overlay
        draw_img = cv2.addWeighted(draw_img, alpha, img, 1 - alpha, 0)

    return draw_img



