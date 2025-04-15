import os
from PIL import Image, ImageFilter
import cv2
import numpy as np


# Using: processed_image_paths = Process_Images(image_paths)
def Process_Images(image_paths: list) -> list:
    """
    Улучшает изображение для лучшего распознавания границ, контуров и углов.
    Возвращает: processed_paths(list): пути скоректированных изображений
    """
    os.makedirs("images/processed", exist_ok=True)
    processed_paths = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        adjusted = cv2.convertScaleAbs(img, alpha=1, beta=30)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        sharpened = img.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=4))
        sharpened_np = np.array(sharpened)
        output_path = os.path.join("images/processed", f"{os.path.basename(image_path)}")
        cv2.imwrite(output_path, cv2.cvtColor(sharpened_np, cv2.COLOR_RGB2BGR))
        processed_paths.append(output_path)
    return processed_paths

# Using: grayscale_image_paths = GrayScale_Images(image_paths)
def GrayScale_Images(image_paths):
    grayscale_paths = []
    output_dir = "images/grayscales"
    os.makedirs(output_dir, exist_ok=True)
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, gray_img)

            grayscale_paths.append(save_path)

    return grayscale_paths



# Using: depthmap_image_paths = DepthMap_Images(image_paths)
def DepthMap_Images(image_paths: list) -> list:

    # output_dir = "images/depthMaps"
    # window_size = 14
    # min_disp = 32
    # nDispFactor = 14
    # num_disp =  abs(16*nDispFactor-min_disp)


    # os.makedirs(output_dir, exist_ok=True)
    # imgLeft = cv2.imread(os.path.join('.', image_paths[0]), cv2.IMREAD_GRAYSCALE)
    # imgRight = cv2.imread(os.path.join('.', image_paths[1]), cv2.IMREAD_GRAYSCALE)

    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=min_disp,
    #     numDisparities=num_disp,
    #     blockSize=window_size,
    #     P1=8*3*window_size*2,
    #     P2=32*3*window_size**2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=33,
    #     speckleWindowSize=0,
    #     speckleRange=2,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )

    # disparity = stereo.compute(imgLeft, imgRight).astype(np.float32) / 16.0

    # disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    # disp_uint8 = np.uint8(disp_norm)
    # save_path = os.path.join('images\\depthMaps', 'depth_map.png')
    # cv2.imwrite(save_path, disp_uint8)

    # return disparity, save_path

    output_dir = "images/depthMaps"
    os.makedirs(output_dir, exist_ok=True)

    if len(image_paths) < 2:
        raise ValueError("Нужно минимум 2 изображения!")

    window_size = 30
    min_disp = 16 # 32
    num_disp = abs(16 * 14 - min_disp)
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize= 33, # window_size,
        P1=8 * 3 * window_size * 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    total_disp = None
    valid_pairs = 0

    for i in range(len(image_paths) - 1):
        img_left = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(image_paths[i + 1], cv2.IMREAD_GRAYSCALE)

        if img_left is None or img_right is None:
            continue

        disp = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

        if total_disp is None:
            total_disp = np.zeros_like(disp)
        total_disp += disp
        valid_pairs += 1

    if valid_pairs == 0:
        raise ValueError("Не удалось обработать ни одной пары изображений!")

    avg_disp = total_disp / valid_pairs
    disp_norm = cv2.normalize(avg_disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_norm)

    save_path = os.path.join(output_dir, "depth_map_final.png")
    cv2.imwrite(save_path, disp_uint8)

    return avg_disp, save_path


