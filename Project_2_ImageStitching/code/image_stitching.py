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
config = {
    "dirname" : "../parrington",
    "left_to_right" : False,
    "pano_save_path" : "./pre_panorama.jpg",
}

#%%
'''
Load images and focals
'''
def get_focals_autostitch(path):
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
            focals = get_focals_autostitch(filepath)
    
    return np.array(rgbs), np.array(focals)

#%%
'''
Store images
'''
def save_image(rgb, path):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

#%%
images, focals = read_data(config["dirname"])

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
def compute_corner_response(gray, kernel=5, sigma=3, k=0.04):
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

    print(f'Rmax: {np.max(R)}\nRmin: {np.min(R)}')
    return R, Ix, Iy, Ix2, Iy2

def get_local_max_R(R, Rthres=0.06):
    localMax = np.ones(R.shape, dtype=np.uint8)
    if np.max(R) > 600000:
        localMax[R <= np.max(R) * 0.03] = 0
    else:    
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

    print(f'Corners: {np.sum(localMax)}')
    feature_points = np.where(localMax > 0)
    return feature_points[1], feature_points[0] # fpx, fpy

def get_orientations(Ix, Iy, Ix2, Iy2, fpx, fpy, bins=36, kernel=9):
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
        votes[b] = cv2.GaussianBlur(votes[b], (kernel, kernel), 1.5)

    ori = np.ones((len(fpx), 2)) * (-1)

    for i in range(len(fpy)):
        y, x = fpy[i], fpx[i]
        vote = votes[:, y, x]
        best_bin_1, best_mag_1 = np.argmax(vote), np.max(vote) # best orientation
        voet_2 = vote.copy()
        voet_2[best_bin_1] = np.min(vote)
        best_bin_2, best_mag_2 = np.argmax(voet_2), np.max(voet_2) # second best orientation
        if best_mag_2 >= best_mag_1 * 0.8:
            ori[i, 0], ori[i, 1] = best_bin_1, best_bin_2
        else:
            ori[i, 0] = best_bin_1

    return ori, M, theta

def get_descriptors(fpx, fpy, ori, M, theta, bins=8):
    
    h, w = M.shape
    half_w = w / 2

    kernel_1d = cv2.getGaussianKernel(16, 16 * 0.5)
    kernel_2d = kernel_1d @ kernel_1d.T
    gaussian_filter = kernel_2d / kernel_2d.sum()
    
    assert(360 % bins == 0)
    bin_size = 360 / bins

    left_descriptors = []
    right_descriptors = []
    for i in range(len(fpy)):
        y, x = fpy[i], fpx[i]
        if y - 8 < 0 or y + 8 >= h or x - 8 < 0 or x + 8 >= w:
            # print(f'Skip Keypoint (y, x) = {y, x}')
            continue
        for j in range(2): # best and second best orientations
            if ori[i][j] == -1: continue # no second best
            vector = []
            local_M = M[y-8:y+8, x-8:x+8]
            local_theta = theta[y-8:y+8, x-8:x+8]
            local_theta_vote_bin = np.floor((local_theta + (bin_size / 2)) / bin_size) % bins
            
            for y_t in range(0, 16, 4):
                for x_t in range(0, 16, 4):
                    sv = np.zeros(bins)
                    for y_offset in range(4):
                        for x_offset in range(4):
                            b = local_theta_vote_bin[y_t+y_offset][x_t+x_offset]
                            sv[int(b)] = local_M[y_t+y_offset][x_t+x_offset] * gaussian_filter[y_t+y_offset][x_t+x_offset]
                        
                    sv_n1 = [x / (np.sum(sv) + 1e-8) for x in sv]
                    sv_clip = [x if x < 0.2 else 0.2 for x in sv_n1]
                    sv_n2 = [x / (np.sum(sv_clip) + 1e-8) for x in sv_clip]

                    vector += sv_n2

            if x < half_w:
                left_descriptors.append({'x': x, 'y': y, 'vector': vector})
            else:
                right_descriptors.append({'x': x, 'y': y, 'vector': vector})


    print(f'Descriptors: {len(left_descriptors) + len(right_descriptors)}')
    return left_descriptors, right_descriptors

#%%
L_DES, R_DES = [], []
for i, image in enumerate(cw_images):
    print(i)
    w_imgs_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    R, Ix, Iy, Ix2, Iy2 = compute_corner_response(w_imgs_gray)
    fpx, fpy = get_local_max_R(R)
    ori, M, theta = get_orientations(Ix, Iy, Ix2, Iy2, fpx, fpy)
    l_des, r_des = get_descriptors(fpx, fpy, ori, M, theta)
    L_DES.append(l_des)
    R_DES.append(r_des)

#%%
def find_matches(des1, des2, thres=0.8):
    des1_vectors = pd.DataFrame(des1)['vector'].tolist()
    des2_vectors = pd.DataFrame(des2)['vector'].tolist()

    distances = cdist(des1_vectors, des2_vectors)
    sorted_index = np.argsort(distances, axis=1)
    matches = []
    for i, si in enumerate(sorted_index):
        first_match = distances[i, si[0]]
        second_match = distances[i, si[1]]
        if (first_match / second_match) < thres:
            matches.append([i, si[0]])
    
    print(f'Matches: {len(matches)}')
    return matches

def ransac(matches, des1, des2, n=1, K=1000):
    matches = np.array(matches)
    m1, m2 = matches[:, 0], matches[:, 1]
    
    P1 = np.array(pd.DataFrame(des1).loc[m1][['x', 'y']])
    P2 = np.array(pd.DataFrame(des2).loc[m2][['x', 'y']])

    Err, Dxy = [], []
    for _ in range(K):
        samples = np.random.randint(0, len(P1), n)
        dxy = np.mean(P1[samples] - P2[samples], axis=0).astype(np.int32)
        diff_xy = np.abs(P1 - (P2 + dxy))
        err = np.sum(np.sign(np.sum(diff_xy, axis=1)))
        Err += [err]
        Dxy += [dxy]

    E_sortindex = np.argsort(Err)
    best_dxy = Dxy[E_sortindex[0]]
    
    return best_dxy

#%%
Dxy = [np.zeros(2).astype(np.int32)]
if config["left_to_right"]:
    for i in range(len(cw_images)-1):
        print(i)
        matches = find_matches(L_DES[i+1], R_DES[i])
        Dxy += [ransac(matches, L_DES[i+1], R_DES[i])]
else:
    for i in range(len(cw_images)-1):
        print(i)
        matches = find_matches(L_DES[i], R_DES[i+1])
        Dxy += [ransac(matches, L_DES[i], R_DES[i+1])]

#%%
def get_panorama_init(Dxy_all, image_shape):
    h, w, c = image_shape
    
    Dx, Dy = Dxy_all[:, 0], Dxy_all[:, 1]
    dx_max, dx_min = np.max(Dx), np.min(Dx)
    dy_max, dy_min = np.max(Dy), np.min(Dy)
    
    offset_x = -min(dx_min, 0)
    offset_y = -min(dy_min, 0)
    
    pano_w = w + (dx_max - dx_min)
    pano_h = h + (dy_max - dy_min)
    
    pano = np.zeros((pano_h, pano_w, c)).astype(np.float32)
    
    return pano, offset_x, offset_y

def blend_linear_helper(pano, im, x, y, sign):
    h, w, c = im.shape
    if np.sum(pano) == 0:
        pano[y:y+h, x:x+w] = im.astype(np.float32)
    else:
        w_pano = np.sign(pano)
    
        w_im = np.zeros(pano.shape)
        w_im[y:y+h, x:x+w][im > 0] = 1
        
        blend_area = w_pano + w_im - np.sign(w_pano + w_im)
        
        sum_x = np.sum(blend_area, axis=0)
        sum_y = np.sum(blend_area, axis=1)
        
        index_x = np.where(sum_x > 0)[0]
        start_x, end_x = index_x[0], index_x[-1] + 1
        index_y = np.where(sum_y > 0)[0]
        start_y, end_y = index_y[0], index_y[-1] + 1

        xlen = end_x - start_x
        ylen = end_y - start_y
        
        inter = np.zeros((ylen, xlen))
        if sign >= 0:
            inter += np.linspace(0, 1, xlen)
        else:
            inter += np.linspace(1, 0, xlen)
            
        inter = np.stack([inter, inter, inter], axis=2)
        
        im_add = np.zeros(pano.shape).astype(np.float32)
        im_add[y:y+h, x:x+w] = im.astype(np.float32)
        
        w_im[start_y:end_y, start_x:end_x] *= inter
        w_im[pano == 0] = 1
        w_im[im_add == 0] = 0
        
        w_pano[start_y:end_y, start_x:end_x] *= (1. - inter)
        w_pano[im_add == 0] = 1
        w_pano[pano == 0] = 0
        
        pano = w_pano * pano + w_im * im_add
        
    return pano

def blend_linear(images, Dxy):
    _, h, w, c = images.shape
    Dxy_all = np.cumsum(Dxy, axis=0)
    
    pano, offset_x, offset_y = get_panorama_init(Dxy_all, (h, w, c))
    
    for _, (image, dxy_all, dxy) in enumerate(zip(images, Dxy_all, Dxy)):
        dx_all, dy_all = dxy_all
        pano = blend_linear_helper(pano, image, dx_all+offset_x, dy_all+offset_y, dxy[0])

    return pano.astype(np.uint8)

#%%
pano = blend_linear(cw_images, Dxy)

#%%
save_image(pano, config["pano_save_path"])
print(f'Pano saved at {config["pano_save_path"]}')

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
    # plt.savefig(f'features-{i}.png')
    plt.show()
#%%
plot_features(cw_images[0], R, fpx, fpy, Ix, Iy)