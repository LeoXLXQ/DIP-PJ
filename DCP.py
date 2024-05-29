import cv2
import numpy as np
import os
from scipy.ndimage import minimum_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.color import rgb2lab, deltaE_ciede2000
import openpyxl
import time

# 计算雾化图像的暗通道
def dark_channel(im, patch_size=15):
    dark = minimum_filter(im, patch_size, mode='nearest')
    dark = np.min(dark, axis=2)
    return dark

# 估计全局大气光值
def get_atmo(img, percent=0.001):
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    sorted_mean = np.sort(mean_perpix)
    num_top_pixels = int(img.shape[0] * img.shape[1] * percent)
    mean_topper = sorted_mean[-num_top_pixels:]
    return np.mean(mean_topper)

# 计算透射率图
def get_trans(img, atom, w=0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t

# 引导滤波
def guided_filter(input_image, guide_image, radius, regularization):
    mean_guide = np.mean(guide_image)
    mean_input = np.mean(input_image)
    corr_guide = np.mean(guide_image * guide_image)
    corr_guide_input = np.mean(guide_image * input_image)
    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_input = corr_guide_input - mean_guide * mean_input
    a = cov_guide_input / (var_guide + regularization)
    b = mean_input - a * mean_guide
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    filtered_image = mean_a * guide_image + mean_b
    return filtered_image

def find_corresponding_result_image_name(gt_image_name, result_image_names):
    gt_prefix = gt_image_name[:2]
    for result_name in result_image_names:
        result_prefix = result_name[:2]
        if gt_prefix == result_prefix:
            return result_name
    return None

def dehaze(originPath, savePath, gtPath):
    start_time = time.time()  # Start time

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.append(["Image", "MSE", "SSIM", "PSNR", "CIEDE2000"])
    
    for image_name in os.listdir(originPath):
        image_path = os.path.join(originPath, image_name)
        im = cv2.imread(image_path)
        
        if im is None:
            print(f"Error loading image {image_path}")
            continue
        
        img = im.astype('float64') / 255
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
        atom = get_atmo(img)
        trans = get_trans(img, atom)
        trans_guided = guided_filter(trans, img_gray, 15, 0.0001)
        trans_guided = np.maximum(trans_guided, 0.25)
        result = np.empty_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
        oneSave = os.path.join(savePath, image_name)
        cv2.imwrite(oneSave, (result * 255).astype(np.uint8))
    
    end_time = time.time()  # End time
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:.2f} seconds")
    
    for image_name in os.listdir(gtPath):
        gt_image_path = os.path.join(gtPath, image_name)
        gt_image = cv2.imread(gt_image_path).astype('float64') / 255
        
        if gt_image is None:
            print(f"Error loading ground truth image {gt_image_path}")
            continue

        corresponding_result_name = find_corresponding_result_image_name(image_name, os.listdir(savePath))
        if corresponding_result_name is None:
            print(f"No corresponding result image found for {image_name}")
            continue

        corresponding_result_path = os.path.join(savePath, corresponding_result_name)
        result = cv2.imread(corresponding_result_path).astype('float64') / 255

        if result is None:
            print(f"Error loading result image {corresponding_result_path}")
            continue

        if gt_image.shape != result.shape:
            result = cv2.resize(result, (gt_image.shape[1], gt_image.shape[0]))

        mse_value = mse(gt_image, result)
        min_dim = min(gt_image.shape[0], gt_image.shape[1])
        win_size = min(min_dim, 3) // 2 * 2 + 1
        ssim_value = ssim(gt_image, result, win_size=win_size, channel_axis=2, data_range=1)
        psnr_value = psnr(gt_image, result, data_range=1)
        gt_lab = rgb2lab(gt_image)
        result_lab = rgb2lab(result)
        ciede2000_value = np.mean(deltaE_ciede2000(gt_lab, result_lab))
        
        sheet.append([image_name, mse_value, ssim_value, psnr_value, ciede2000_value])
    
    excel_file = os.path.join(savePath, "evaluation_results.xlsx")
    wb.save(excel_file)



if __name__ == '__main__':
    dehaze('/Users/xl/Desktop/NIP/NHHaze/hazy', '/Users/xl/Desktop/NIP/NHHaze/DCP', '/Users/xl/Desktop/NIP/NHHaze/GT')
