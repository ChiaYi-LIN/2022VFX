#%%
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
from scipy.spatial.distance import cdist

#%%
_dirname_ = '../parrington'

#%%
'''
Load images and focals
'''
def get_focals(path):
    focals = []
    with open(path, 'r') as f:
        data = [line for line in f]
        for i in range(1, len(data)-1):
            if data[i-1] == '\n' and data[i+1] == '\n':
                focals += [float(data[i])]
    return focals

def read_data(dirname, max_h=480, img_extensions=['.png', '.jpg', '.gif', '.JPG'], ignore_filenames=['pano.jpg']):    
    rgbs, focals = [], []
    filenames = sorted(os.listdir(dirname))
    # print(filenames)
    for filename in filenames:
        if filename in ignore_filenames: continue
        name, extension = osp.splitext(filename)
        filepath = osp.join(dirname, filename)
        if extension in img_extensions:
            bgr = cv2.imread(filepath)
            h, w, c = bgr.shape
            if h > max_h:
                new_w = int(w * (max_h / h))
                bgr = cv2.resize(bgr, (new_w, max_h), cv2.INTER_LINEAR)
            rgbs += [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)]
        elif extension in ['.txt']:
            focals = get_focals(filepath)
    
    return np.array(rgbs), np.array(focals)

#%%
'''
Store images
'''
def save_image(rgb, path):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

#%%
images, focals = read_data(_dirname_)

#%%
def cylinder_warping(imgs, focals):
    n, h, w, c = imgs.shape
    warped_imgs = np.zeros([n, h, w, c])
    x_origin = np.floor(w / 2)
    y_origin = np.floor(h / 2)
    for i in range(n):
        s = float(focals[i])
        for y_new in range(h):
            y_prime = y_new - y_origin
            for x_new in range(w):
                x_prime = x_new - x_origin
                x = s * np.tan(x_prime / s)
                y = np.sqrt(x**2 + s**2) * y_prime / s
                x = int(np.round(x + x_origin))
                y = int(np.round(y + y_origin))
                if x >= 0 and x < w and y >= 0 and y < h:
                    warped_imgs[i, y_new, x_new, :] = imgs[i, y, x, :]

    return warped_imgs.astype(np.uint8)  

#%%
cw_images = cylinder_warping(images, focals)

#%%
def compute_corner_response(gray, kernel=9, sigma=3, k=0.04):
    K = (kernel, kernel)

    # Compute x and y derivatives of image
    gray_blur = cv2.GaussianBlur(gray, K, sigma)
    Iy, Ix = np.gradient(gray_blur)

    # Compute products of derivatives at every pixel
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    # Compute the sums of the products of derivatives at each pixel
    Sx2 = cv2.GaussianBlur(Ix2, K, sigma)
    Sy2 = cv2.GaussianBlur(Iy2, K, sigma)
    Sxy = cv2.GaussianBlur(Ixy, K, sigma)

    # Compute the response of the detector at each pixel
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2
    R = detM - k * (traceM ** 2)

    print(f'max R: {np.max(R)}, min R: {np.min(R)}')
    return R, Ix, Iy, Ix2, Iy2

def get_local_max_R(R, Rthres=0.06):
    localMax = np.ones(R.shape, dtype=np.uint8)
    localMax[R <= np.max(R) * Rthres] = 0

    kernel = np.ones((3,3), np.uint8)
    kernel[1,1] = 0
    R_dilation = cv2.dilate(R, kernel)
    
    for i in range(localMax.shape[0]):
        for j in range(localMax.shape[1]):
            if localMax[i, j] == 1 and R[i, j] > R_dilation[i, j]:
                localMax[i, j] = 1
            else:
                localMax[i, j] = 0

    print(f'Number of Corners: {np.sum(localMax)}')
    feature_points = np.where(localMax > 0)
    return feature_points[1], feature_points[0] # fpx, fpy

def get_orientations(Ix, Iy, Ix2, Iy2, bins=36, kernel=9):
    M = (Ix2 + Iy2) ** (1/2)
    theta = np.arctan2(Iy, Ix) * (180 / np.pi)
    theta = (theta + 360) % 360

    assert(360 % bins == 0)
    bin_size = 360 / bins
    theta_vote_bin = np.floor((theta + (bin_size / 2)) / bin_size) % bins

    votes = np.zeros((bins,) + Ix.shape)
    for b in range(bins):
        votes[b][theta_vote_bin == b] = 1
        votes[b] *= M
        votes[b] = cv2.GaussianBlur(votes[b], (kernel, kernel), 0)

    ori = np.argmax(votes, axis=0)

    return ori, M, theta

def get_descriptors(fpx, fpy, ori, M, theta, ori_bin_size=10, bins=8):
    
    h, w = M.shape
    half_w = w / 2
    assert(360 % bins == 0)
    bin_size = 360 / bins

    left_descriptors = []
    right_descriptors = []
    for y, x in zip(fpy, fpx):
        if y - 12 < 0 or y + 12 >= h or x - 12 < 0 or x + 12 >= w:
            # print(f'Skip Keypoint (y, x) = {y, x}')
            continue
        vector = []
        rotation = ori[y, x] * ori_bin_size
        rotation_matrix = cv2.getRotationMatrix2D((12, 12), rotation, 1) # cv2.getRotationMatrix2D(center, angle, scale) 
        rotated_M = cv2.warpAffine(M[y-12:y+12, x-12:x+12], rotation_matrix, (24, 24))
        rotated_theta = (theta[y-12:y+12, x-12:x+12] - rotation + 360) % 360
        rotated_theta_vote_bin = np.floor((rotated_theta + (bin_size / 2)) / bin_size) % bins
        for fy in range(4, 20, 4):
            for fx in range(4, 20, 4):
                sv = np.zeros(bins)
                for y_offset in range(4):
                    for x_offset in range(4):
                        b = rotated_theta_vote_bin[fy + y_offset][fx + x_offset]
                        sv[int(b)] = rotated_M[fy + y_offset][fx + x_offset]
                    
                sv_n1 = [x / (np.sum(sv) + 1e-8) for x in sv]
                sv_clip = [x if x < 0.2 else 0.2 for x in sv_n1]
                sv_n2 = [x / (np.sum(sv_clip) + 1e-8) for x in sv_clip]

                vector += sv_n2

        if x < half_w:
            left_descriptors.append({'y': y, 'x': x, 'vector': vector})
        else:
            right_descriptors.append({'y': y, 'x': x, 'vector': vector})
    
    print(f'Number of Descriptors: Left = {len(left_descriptors)}, Right = {len(right_descriptors)}')
    return left_descriptors, right_descriptors

#%%
L_DES, R_DES = [], []
for i, image in enumerate(cw_images):
    print(f"Processing image {i}")
    w_imgs_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    R, Ix, Iy, Ix2, Iy2 = compute_corner_response(w_imgs_gray)
    fpx, fpy = get_local_max_R(R)
    ori, M, theta = get_orientations(Ix, Iy, Ix2, Iy2)
    l_des, r_des = get_descriptors(fpx, fpy, ori, M, theta)
    L_DES.append(l_des)
    R_DES.append(r_des)

#%%


#%%


#%%
l_Des += [l_des]
r_Des += [r_des]

#%%
theta.shape

#%%
'''
Plot
'''
def plot_features(im, R, fpx, fpy, Ix, Iy, arrow_size=1.0, i=0):
    h, w, c = im.shape
    
    feature_points = np.copy(im)
    for x, y in zip(fpx, fpy):
        cv2.circle(feature_points, (x, y), radius=1, color=[255, 0, 0], thickness=1, lineType=1) 
        
    feature_gradients = np.copy(im)
    for x, y in zip(fpx, fpy):
        ex, ey = int(x + Ix[y, x] / arrow_size), int(y + Iy[y, x] / arrow_size)
        ex, ey = np.clip(ex, 0, w), np.clip(ey, 0, h)
        cv2.arrowedLine(feature_gradients, (x, y), (ex, ey), (255, 255, 0), 1)
        
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].imshow(im); ax[0, 0].set_title('Original')
    ax[0, 1].imshow(np.log(R + 1e-4), cmap='jet'); ax[0, 1].set_title('R')
    ax[1, 0].imshow(feature_points); ax[1, 0].set_title('Feature Points')
    ax[1, 1].imshow(feature_gradients); ax[1, 1].set_title('Gradients')
    plt.savefig(f'features-{i}.png')
    # plt.show()
#%%
plot_features(cw_images[0], R, fpx, fpy, Ix, Iy)