import numpy as np


PARKING_LOT_SKELETON = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]


# KINEMATIC_TREE_SKELETON = [
#     (1, 2), (2, 4),  # left head
#     (1, 3), (3, 5),
#     (1, 6),
#     (6, 8), (8, 10),  # left arm
#     (1, 7),
#     (7, 9), (9, 11),  # right arm
#     (6, 12), (12, 14), (14, 16),  # left side
#     (7, 13), (13, 15), (15, 17),
# ]


PARKING_KEYPOINTS = [
    'p1',
    'p2',
    'p3',
    'p4',
]


PARKING_UPRIGHT_POSE = np.array([
    [0,0, 2.0],  # 'nose',            # 1
    [10, 0, 2.0],  # 'left_eye',        # 2
    [10,10, 2.0],  # 'right_eye',       # 3
    [0,10, 2.0],  # 'left_ear',        # 4
])


HFLIP = {
    'p1': 'p3',
    'p2': 'p4',
    'p3':'p1',
    'p4':'p2',
}
#     'left_shoulder': 'right_shoulder',
#     'right_shoulder': 'left_shoulder',
#     'left_elbow': 'right_elbow',
#     'right_elbow': 'left_elbow',
#     'left_wrist': 'right_wrist',
#     'right_wrist': 'left_wrist',
#     'left_hip': 'right_hip',
#     'right_hip': 'left_hip',
#     'left_knee': 'right_knee',
#     'right_knee': 'left_knee',
#     'left_ankle': 'right_ankle',
#     'right_ankle': 'left_ankle',
# }


# DENSER_PARKING_PERSON_SKELETON = [
#     (1, 2), (1, 3), (2, 3), (1, 4), (1, 5), (4, 5),
#     (1, 6), (1, 7), (2, 6), (3, 7),
#     (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
#     (6, 12), (7, 13), (6, 13), (7, 12), (12, 13),
#     (6, 8), (7, 9), (8, 10), (9, 11), (6, 10), (7, 11),
#     (8, 9), (10, 11),
#     (10, 12), (11, 13),
#     (10, 14), (11, 15),
#     (14, 12), (15, 13), (12, 15), (13, 14),
#     (12, 16), (13, 17),
#     (16, 14), (17, 15), (14, 17), (15, 16),
#     (14, 15), (16, 17),
# ]


# DENSER_PARKING_PERSON_CONNECTIONS = [
#     c
#     for c in DENSER_PARKING_PERSON_SKELETON
#     if c not in PARKING_PERSON_SKELETON
# ]


PARKING_LOT_SIGMAS = [
    0.026,  # p1
    0.025,  # p2
    0.025,  # p3
    0.035,  # p4
]


PARKING_LOT_SCORE_WEIGHTS = [3.0] * 3 + [1.0] * (len(PARKING_KEYPOINTS) - 3)


PARKING_CATEGORIES = [
    'parking_lot',
]


def draw_skeletons(pose):
    import openpifpaf  # pylint: disable=import-outside-toplevel
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    keypoint_painter = openpifpaf.show.KeypointPainter()

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    ann = openpifpaf.Annotation(keypoints=PARKING_KEYPOINTS,
                                skeleton=PARKING_LOT_SKELETON,
                                score_weights=PARKING_LOT_SCORE_WEIGHTS)
    ann.set(pose, np.array(PARKING_LOT_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann, filename='docs/skeleton_parking.png') as ax:
        keypoint_painter.annotation(ax, ann)

    # ann_kin = openpifpaf.Annotation(keypoints=PARKING_KEYPOINTS,
    #                                 skeleton=KINEMATIC_TREE_SKELETON,
    #                                 score_weights=PARKING_LOT_SCORE_WEIGHTS)
    # ann_kin.set(pose, np.array(PARKING_LOT_SIGMAS) * scale)
    # with openpifpaf.show.Canvas.annotation(
    #         ann_kin, filename='docs/skeleton_kinematic_tree.png') as ax:
    #     keypoint_painter.annotation(ax, ann_kin)

    # ann_dense = openpifpaf.Annotation(keypoints=PARKING_KEYPOINTS,
    #                                   skeleton=DENSER_PARKING_PERSON_SKELETON,
    #                                   score_weights=PARKING_LOT_SCORE_WEIGHTS)
    # ann_dense.set(pose, np.array(PARKING_LOT_SIGMAS) * scale)
    # with openpifpaf.show.Canvas.annotation(
    #         ann, ann_bg=ann_dense, filename='docs/skeleton_dense.png') as ax:
    #     keypoint_painter.annotation(ax, ann_dense)


def print_associations():
    for j1, j2 in PARKING_LOT_SKELETON:
        print(PARKING_KEYPOINTS[j1 - 1], '-', PARKING_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(PARKING_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(PARKING_UPRIGHT_POSE)
