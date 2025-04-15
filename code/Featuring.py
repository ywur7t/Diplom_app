import cv2
import numpy as np
import matplotlib.pyplot as plt



def extract_keypoints_with_depth(img, disparity, focal_length, baseline, depth_threshold=(0.1, 5.0)):
    """
    Выделяет ключевые точки с учётом карты глубины.

    **Входные параметры**:
    img – изображение,
    disparity – карта смещений,
    focal_length – фокусное расстояние (в пикселях),
    baseline – расстояние между камерами (в метрах),
    depth_threshold – (min, max) глубины в метрах для фильтрации.

    **Переменные**:
    depth_map – карта глубины,
    keypoints – список ключевых точек,
    valid_keypoints – отфильтрованные точки по глубине,
    descriptors – дескрипторы.

    **Выходные параметры**:
    keypoints, descriptors, depth_map – список точек, их дескрипторы, и карта глубины.
    """

    with np.errstate(divide='ignore'):
        depth_map = (focal_length * baseline) / disparity
    depth_map[disparity <= 0.0] = 0


    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(cv2.imread(img, cv2.IMREAD_GRAYSCALE), None)

    points_with_depth = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            d = disparity[y, x]
            if d > 0:
                Z = (focal_length * baseline) / d * 50

                points_with_depth.append((x, y, Z))

    return keypoints, descriptors, points_with_depth


def draw_keypoints_with_depth(img, points_with_depth):

    scale_z = 50
    xs, ys, zs = zip(*[(x, y, z * scale_z) for x, y, z in points_with_depth])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, c=zs, cmap='jet', s=3)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z (depth)')

    ax.view_init(elev=20, azim=-60)
    plt.title("SIFT Keypoints in 3D with Depth")
    plt.show()

