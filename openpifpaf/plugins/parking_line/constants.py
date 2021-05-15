import numpy as np


PARKING_LINE_SKELETON = [[1, 2], [2,1]]

PARKING_KEYPOINTS = [
    'p1',
    'p2',
]


PARKING_POSE = np.array([
    [0,0, 2],  # 'nose',            # 1
    [10, 0, 10.0],  # 'left_eye',        # 2
])


HFLIP = {
    'p1': 'p2',
    'p2': 'p1',
}


PARKING_LINE_SIGMAS = [
    0.026,  # p1
    0.025,  # p2
]

# PARKING_LINE_SCORE_WEIGHTS = [3.0] * 3 + [1.0] * (len(PARKING_KEYPOINTS) - 3)
PARKING_LINE_SCORE_WEIGHTS = [3,3]


PARKING_CATEGORIES = [
    'parking_line',
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
                                skeleton=PARKING_LINE_SKELETON,
                                score_weights=PARKING_LINE_SCORE_WEIGHTS)
    ann.set(pose, np.array(PARKING_LINE_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann, filename='docs/skeleton_parking.png') as ax:
        keypoint_painter.annotation(ax, ann)

def print_associations():
    for j1, j2 in PARKING_LINE_SKELETON:
        print(PARKING_KEYPOINTS[j1 - 1], '-', PARKING_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(PARKING_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(PARKING_POSE)
