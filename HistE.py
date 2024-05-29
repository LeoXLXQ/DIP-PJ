import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.color import rgb2lab, deltaE_ciede2000
import openpyxl
import time

def calculate_histogram(img):
    hist = np.zeros(256, dtype=int)
    height, width = img.shape[:2]
    for row in range(height):
        for col in range(width):
            gray_value = img[row, col]
            hist[gray_value] += 1
    return hist

def draw_histogram(img, title='Histogram'):
    x = np.asarray(img).flatten()
    plt.figure()
    plt.hist(x, bins=256, color='green')
    plt.xlabel('Gray Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

def calculate_cumulative_distribution(hist):
    cdf = np.zeros(256, dtype=float)
    total_pixels = np.sum(hist)
    for i in range(256):
        if i == 0:
            cdf[i] = hist[i] / total_pixels
        else:
            cdf[i] = hist[i] / total_pixels + cdf[i - 1]
    return cdf

def draw_cumulative_distribution(cdf, title='Cumulative Distribution Function'):
    x = list(range(256))
    plt.figure()
    plt.xlim(0, 255)
    plt.ylim(0, 1)
    plt.xlabel('Gray Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.plot(x, cdf * 255, color='green', linewidth=0.5)
    plt.show()

def equalize_image(img):
    channels = cv2.split(img)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)

def draw_image(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        
       #gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #draw_histogram(gray_img, title='Original Image Histogram')
        #hist = calculate_histogram(gray_img)
        #cdf = calculate_cumulative_distribution(hist)
        #draw_cumulative_distribution(cdf, title='Cumulative Distribution Function')
    
        eq_img = equalize_image(im)
        # draw_image(eq_img, title='Equalized Image')
        #eq_gray_img = cv2.cvtColor(eq_img, cv2.COLOR_BGR2GRAY)
        #draw_histogram(eq_gray_img, title='Equalized Image Histogram')

        oneSave = os.path.join(savePath, image_name)
        cv2.imwrite(oneSave, eq_img)
        
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
    dehaze('/Users/xl/Desktop/NIP/NHHaze/hazy', '/Users/xl/Desktop/NIP/NHHaze/HistE', '/Users/xl/Desktop/NIP/NHHaze/GT')

