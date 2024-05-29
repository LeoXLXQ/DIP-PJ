import cv2
import numpy as np
import os
from scipy.ndimage import minimum_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.color import rgb2lab, deltaE_ciede2000
import pandas as pd
import openpyxl
import time

# 绘图
def display_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 暗通道
def get_minimum_channel(image):
    return np.min(image, axis=2)
    
# 白平衡处理
def apply_white_balance(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    red_avg, green_avg, blue_avg = cv2.mean(red_channel)[0], cv2.mean(green_channel)[0], cv2.mean(blue_channel)[0]
    gray_avg = (red_avg + green_avg + blue_avg) / 3
    k_red, k_green, k_blue = gray_avg / red_avg, gray_avg / green_avg, gray_avg / blue_avg
    red_channel = cv2.addWeighted(red_channel, k_red, 0, 0, 0)
    green_channel = cv2.addWeighted(green_channel, k_green, 0, 0, 0)
    blue_channel = cv2.addWeighted(blue_channel, k_blue, 0, 0, 0)
    return cv2.merge([blue_channel, green_channel, red_channel])

# 去雾
def dehaze_image(original_image, smooth_value=3, percentage=0.95):
    white_balanced_image = apply_white_balance(original_image)
    min_channel_image = get_minimum_channel(white_balanced_image)
    
    atmospheric_light = cv2.medianBlur(np.uint8(min_channel_image), smooth_value)
    light_difference = np.abs(min_channel_image - atmospheric_light)
    refined_difference = atmospheric_light - cv2.medianBlur(np.uint8(light_difference), smooth_value)
    
    max_255_image = np.ones(refined_difference.shape, dtype=np.uint8) * 255
    min_transmission_map = cv2.merge([np.uint8(percentage * refined_difference), np.uint8(min_channel_image), max_255_image])
    min_transmission_map = get_minimum_channel(min_transmission_map)
    min_transmission_map[min_transmission_map < 0] = 0

    blurred_map = cv2.blur(np.uint8(min_transmission_map), (5, 5))

    normalized_map = np.float32(blurred_map) / 255
    dehazed_image = np.zeros((normalized_map.shape[0], normalized_map.shape[1], 3), dtype=np.float32)
    normalized_white_balanced_image = np.float32(white_balanced_image) / 255

    for i in range(3):
        dehazed_image[:, :, i] = (normalized_white_balanced_image[:, :, i] - normalized_map) / (1 - normalized_map)
    dehazed_image = np.clip(dehazed_image / dehazed_image.max(), 0, 1)
    dehazed_image = np.uint8(dehazed_image * 255)

    return dehazed_image

# 保存
def save_dehazed_image(file_path, dehazed_image):
    cv2.imwrite(file_path, dehazed_image)

# 对比
def find_matching_result_image_name(gt_image_name, result_image_names):
    gt_prefix = gt_image_name[:4]
    for result_name in result_image_names:
        result_prefix = result_name[:4]
        if gt_prefix == result_prefix:
            return result_name
    return None
    
def dehaze(hazy_image_folder, dehazed_image_folder, ground_truth_folder):
    start_time = time.time()  # Start time

    if not os.path.exists(dehazed_image_folder):
        os.makedirs(dehazed_image_folder)
        
    hazy_image_list = os.listdir(hazy_image_folder)
    if not hazy_image_list:
        print("No files found in the input folder.")
        return
    
    for image_name in hazy_image_list:
        file_path = os.path.join(hazy_image_folder, image_name)
        if os.path.isfile(file_path):
            original_image = cv2.imread(file_path)
            if original_image is None:
                print(f"Failed to read image {file_path}")
                continue
            
            dehazed_image = dehaze_image(original_image)
            output_file_path = os.path.join(dehazed_image_folder, image_name)
            save_dehazed_image(output_file_path, dehazed_image)
        
    end_time = time.time()  # End time
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:.2f} seconds")
            
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Image", "MSE", "SSIM", "PSNR", "CIEDE2000"])
    
    for image_name in os.listdir(ground_truth_folder):
        gt_image_path = os.path.join(ground_truth_folder, image_name)
        gt_image = cv2.imread(gt_image_path).astype('float32') / 255
        
        if gt_image is None:
            print(f"Error loading ground truth image {gt_image_path}")
            continue


        matching_result_name = find_matching_result_image_name(image_name, os.listdir(dehazed_image_folder))
        if matching_result_name is None:
            print(f"No corresponding result image found for {image_name}")
            continue

        result_image_path = os.path.join(dehazed_image_folder, matching_result_name)
        result_image = cv2.imread(result_image_path)

        if result_image is None:
            print(f"Error loading result image {result_image_path}")
            continue
        
        result_image = result_image.astype('float32') / 255

        if gt_image.shape != result_image.shape:
            print(f"Resizing result image {result_image_path} to match ground truth image {gt_image_path}.")
            result_image = cv2.resize(result_image, (gt_image.shape[1], gt_image.shape[0]))

        mse_value = mse(gt_image, result_image)
        min_dim = min(gt_image.shape[0], gt_image.shape[1])
        win_size = min(min_dim, 3) // 2 * 2 + 1
        ssim_value = ssim(gt_image, result_image, win_size=win_size, channel_axis=2, data_range=1)
        psnr_value = psnr(gt_image, result_image, data_range=1)
        gt_lab = rgb2lab(gt_image)
        result_lab = rgb2lab(result_image)
        ciede2000_value = np.mean(deltaE_ciede2000(gt_lab, result_lab))
        
        sheet.append([image_name, mse_value, ssim_value, psnr_value, ciede2000_value])
    
    excel_file = os.path.join(dehazed_image_folder, "evaluation_results.xlsx")
    workbook.save(excel_file)


if __name__ == '__main__':
    dehaze('/Users/xl/Desktop/NIP/NHHaze/hazy', '/Users/xl/Desktop/NIP/NHHaze/Fast', '/Users/xl/Desktop/NIP/NHHaze/GT')

