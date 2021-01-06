import numpy as np
from scipy.spatial.transform import Rotation as R


def project_cube_xy_plane(orientation):
    rot = R.from_quat(orientation)
    axes = np.eye(3)
    axes_rotated = rot.apply(axes)

    # calculate the angle between each rotated axis and xy plane
    cos = axes_rotated[:, 2]  # dot product with z_axis

    # choose the nearest axis to z_axis
    # HACK
    # CANT CHOOSE THE SECOND AXIS FOR THE CUBOID
    # IF YOU DO IT THE CUBOID STANDS UPRIGHT, WHICH WON'T HAPPEN ON THE REAL ROBOT
    idx = np.argmax(np.abs(cos)[np.array([0, 2])])
    if idx == 1:
        idx = 2
    sign = np.sign(cos[idx])

    # calculate align rotation
    rot_align = vector_align_rotation(axes_rotated[idx], sign * axes[2])

    return (rot_align * rot).as_quat()


def vector_align_rotation(a, b):
    """
    return Rotation that transform vector a to vector b

    input
    a : np.array(3)
    b : np.array(3)

    return
    rot : scipy.spatial.transform.Rotation
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    assert norm_a != 0 and norm_b != 0

    a = a / norm_a
    b = b / norm_b

    cross = np.cross(a, b)
    norm_cross = np.linalg.norm(cross)
    cross = cross / norm_cross
    dot = np.dot(a, b)

    if norm_cross < 1e-8 and dot > 0:
        '''same direction, no rotation a == b'''
        return R.from_quat([0, 0, 0, 1])
    elif norm_cross < 1e-8 and dot < 0:
        '''opposite direction a == -b'''
        c = np.eye(3)[np.argmax(np.linalg.norm(np.eye(3) - a, axis=1))]
        cross = np.cross(a, c)
        norm_cross = np.linalg.norm(cross)
        cross = cross / norm_cross

        return R.from_rotvec(cross * np.pi)

    rot = R.from_rotvec(cross * np.arctan2(norm_cross, dot))

    assert np.linalg.norm(rot.apply(a) - b) < 1e-7
    return rot


def pitch_rotation_times(cube_orientation, goal_orientation):
    rot_cube = R.from_quat(cube_orientation)
    rot_goal = R.from_quat(goal_orientation)

    rot = rot_goal * rot_cube.inv()

    z_axis = np.array([0, 0, 1])

    z_axis_rotated = rot.apply(z_axis)

    cos_z = z_axis.dot(z_axis_rotated)

    if cos_z > np.cos(np.pi / 4):
        return 0
    elif cos_z > np.cos(np.pi * 3 / 4):
        return 1
    else:
        return 2


def align_z(cube_orientation, projected_goal_orientation):
    pitch_times = pitch_rotation_times(cube_orientation,
                                       projected_goal_orientation)
    rot = R.from_quat(projected_goal_orientation)
    rot_cube = R.from_quat(cube_orientation)

    if pitch_times == 0:
        return rot.as_quat(), 'z', 0

    if pitch_times == 1:
        axes = ['x', 'x', 'y', 'y']
        angles = [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2]

        rot_aligns = [R.from_euler(axis, angle)
                      for axis, angle in zip(axes, angles)]
        rot_diff = [(rot * rot_align * rot_cube.inv())
                    for rot_align in rot_aligns]
        expmap = [rot.as_rotvec() for rot in rot_diff]
        norm_expmap_xy = [np.linalg.norm(vec[:2]) for vec in expmap]

        idx = np.argmin(norm_expmap_xy)

        return (rot * rot_aligns[idx]).as_quat(), axes[idx], -angles[idx]

    if pitch_times == 2:
        axes = ['x', 'y']

        rot_aligns = [R.from_euler(axis, np.pi) for axis in axes]
        diff_mag = [(rot * rot_align * rot_cube.inv()).magnitude()
                    for rot_align in rot_aligns]

        idx = np.argmin(diff_mag)

        return (rot * rot_aligns[idx]).as_quat(), axes[idx], np.pi


def pitch_rotation_axis_and_angle(cube_tip_positions):
    margin = 0.3  # rotate a bit more than 90 degrees.
    x_mean = np.mean(np.abs(cube_tip_positions[:, 0]))
    y_mean = np.mean(np.abs(cube_tip_positions[:, 1]))
    if x_mean > y_mean:
        rotate_axis = "x"
    else:
        rotate_axis = "y"

    if rotate_axis == "x":
        idx = np.argmax(np.abs(cube_tip_positions[:, 1]))
        if cube_tip_positions[idx, 1] > 0:
            rotate_angle = np.pi / 2 * (1 + margin)
        else:
            rotate_angle = -np.pi / 2 * (1 + margin)
    else:
        idx = np.argmax(np.abs(cube_tip_positions[:, 0]))
        if cube_tip_positions[idx, 0] > 0:
            rotate_angle = -np.pi / 2 * (1 + margin)
        else:
            rotate_angle = np.pi / 2 * (1 + margin)

    return rotate_axis, rotate_angle
