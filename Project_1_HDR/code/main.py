#%%
from PIL import Image
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

#%%
_img_dir_ = '../data'
_file_type_ = 'JPG'

num_photos = 7
ref_idx = 3 # int(len(image_c1_list) / 2)

#%%
####################################
# Load images
####################################

#%%
image_RGB_list = []
image_YCbCr_list = []
for i, filename in enumerate(sorted(glob.glob(_img_dir_+'/original/*.'+_file_type_))): #assuming gif
    if i == num_photos:
        break
    print(f'image {i}: {filename}')
    im = Image.open(filename)
    image_RGB_list.append(im.convert('RGB'))
    image_YCbCr_list.append(im.convert('YCbCr'))

# np.array(image_RGB_list[ref_idx].getchannel('G'))

#%%
# Resize image
def resizeImage(image_list=None, set_height=512):
    if image_list is None:
        return image_list
    width, height = image_list[0].size
    if set_height < height:
        new_height = set_height
        new_width = int(width * (set_height / height))
        new_image_list = []
        for img in image_list:
            new_image_list.append(img.resize((new_width, new_height), Image.ANTIALIAS))

        return new_width, new_height, new_image_list
    else:
        return width, height, image_list

#%%
# height, width, _ = np.array(image_RGB_list[0]).shape # no resize
width, height, image_RGB_list = resizeImage(image_list=image_RGB_list, set_height=1024)
width, height, image_YCbCr_list = resizeImage(image_list=image_YCbCr_list, set_height=1024)

# image_RGB_list[ref_idx].show()

#%%
# Show images
fig = plt.figure(figsize=(56, 18), dpi=100)
for i in range(num_photos):
    sub_fig = fig.add_subplot(2, int(num_photos / 2) + 1, i + 1)
    plt.imshow(image_RGB_list[i])
    plt.axis('off')

path_row_photos = f'{_img_dir_}/combine_photos.jpg'
fig.savefig(path_row_photos)
print(f'Combined photos saved at {path_row_photos}')

#%%
####################################
# MTB Alignment
####################################

#%%
def imToArr(im=None):
    im2arr = np.array(im) # im2arr.shape: height x width x channel
    # print(im2arr.shape)
    im2arr = np.transpose(im2arr, (2, 0, 1)) # im2arr.shape: channel x height x width
    return im2arr

def getEachChannelArr(image_list=None):
    if image_list is None:
        return image_list
    H, W, C = np.array(image_list[0]).shape
    print(f'Channel: {C}, Height: {H}, Weight: {W}')
    c1, c2, c3 = [], [], []
    for im in image_list:
        im2arr = imToArr(im)
        c1.append(im2arr[0])
        c2.append(im2arr[1])
        c3.append(im2arr[2])
    return c1, c2, c3

def intensityToBitmaps(intensity=None, exclusion_tolerance=4):
    arr = np.array(intensity)
    median_value = int(np.median(arr))
    intensity_bitmap = np.where(arr > median_value, 1, 0)
    exclusion_bitmap = np.where((arr <= median_value + exclusion_tolerance) & (arr >= median_value - exclusion_tolerance), 0, 1)
    return intensity_bitmap, exclusion_bitmap

def intensitiesToBitmaps(intensity_list=None, exclusion_tolerance=4):
    intensity_bitmap_list = []
    exclusion_bitmap_list = []
    for arr in intensity_list:
        intensity_bitmap, exclusion_bitmap = intensityToBitmaps(arr)
        intensity_bitmap_list.append(intensity_bitmap)
        exclusion_bitmap_list.append(exclusion_bitmap)
    return intensity_bitmap_list, exclusion_bitmap_list

def bitmapToImage(bitmap=None, multiplier=255):
    return Image.fromarray(np.uint8(bitmap * multiplier) , 'L')

def shiftArr(X, dx, dy):
    origin_X = X
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

#%%
image_R_list, image_G_list, image_B_list = getEachChannelArr(image_list=image_RGB_list)
image_Y_list, image_Cb_list, image_Cr_list = getEachChannelArr(image_list=image_YCbCr_list)

#%%
# intensity_bitmaps, exclusion_bitmaps = intensitiesToBitmaps(image_Y_list)
# intensity_bitmap_1 = intensity_bitmaps[ref_idx]
# exclusion_bitmap_1 = exclusion_bitmaps[ref_idx]
# bitmapToImage(intensity_bitmap_1, multiplier=255).show()
# bitmapToImage(exclusion_bitmap_1, multiplier=255).show()

#%%
# img_1 = image_Y_list[0]
# img_1 = Image.fromarray(img_1, 'L')
# img_1.resize((int(img_1.size[0] / 2), int(img_1.size[1] / 2))).show()

#%%
# Calculate translation
def align(ref_img, img, base_dx=0, base_dy=0):
    ref_i_bm, ref_e_bm = intensityToBitmaps(ref_img)
    img_i_bm, img_e_bm = intensityToBitmaps(img)
    min_diff = (ref_i_bm != img_i_bm).sum()
    dx, dy = base_dx, base_dy
    for i in range(-1, 2):
        for j in range(-1, 2):
            diff = shiftArr(img_i_bm, base_dx + i, base_dy + j) != ref_i_bm
            exclude = shiftArr(img_e_bm, base_dx + i, base_dy + j) & ref_e_bm
            diff = (diff & exclude).sum()
            # print(f"({i}, {j}), diff = {diff}")
            if min_diff > diff:
                min_diff = diff
                dx = base_dx + i
                dy = base_dy + j
    return dx, dy

def MTB(ref_img=None, img=None):
    dx, dy = 0, 0
    for i in range(5, -1, -1):
        r_width, r_height = int(width / pow(2, i)), int(height / pow(2, i))
        ref_img = ref_img.resize((r_width, r_height))
        img = img.resize((r_width, r_height))
        dx, dy = align(ref_img, img, 2 * dx, 2 * dy)
        # print(f"dx = {dx}, dy = {dy}")
    return dx, dy

translation = []

ref_intensity = image_Y_list[ref_idx]
ref_img = Image.fromarray(ref_intensity, 'L')
for i in range(len(image_Y_list)):
    intensity = image_Y_list[i]
    img = Image.fromarray(intensity, 'L')
    dx, dy = MTB(ref_img=ref_img, img=img)
    translation.append((dx, dy))
print(f'Translation: {translation}')

#%%
# Shift each channel by translation
aligned_channels = []
for image_list in [image_R_list, image_G_list, image_B_list, image_Y_list, image_Cb_list, image_Cr_list]:
    aligned_list = []
    for i in range(len(translation)):
        dx = translation[i][0]
        dy = translation[i][1]
        # print(f'Image {i} shift right {dx} pixels, shift down {dy} pixels')
        aligned = shiftArr(image_list[i], dx, dy)
        aligned_list.append(aligned)
    aligned_channels.append(aligned_list)

aligned_RGB_list = aligned_channels[0:3]
aligned_YCbCr_list = aligned_channels[3:6]
aligned_R_list, aligned_G_list, aligned_B_list = aligned_RGB_list[0], aligned_RGB_list[1], aligned_RGB_list[2]
aligned_Y_list, aligned_Cb_list, aligned_Cr_list = aligned_YCbCr_list[0], aligned_YCbCr_list[1], aligned_YCbCr_list[2]

#%%
####################################
# Recover response curves
####################################

#%%
random.seed(1121326)
sample_points = []
num_samples = 1000
for i in range(num_samples):
    sample_h = random.randint(32, height - 32 - 1)
    sample_w = random.randint(32, width - 32 - 1)
    sample_points.append((sample_h, sample_w))

#%%
log_shutter_time = [-i for i in range(20)]

#%%
# Recover response curve for each channel
g_RGB = []
for aligned_list in aligned_RGB_list:
    A = []
    b = []
    for i in range(num_samples):
        for j in range(num_photos):
            row = [0 for _ in range(256 + num_samples)]
            z_ij = aligned_list[j][sample_points[i][0]][sample_points[i][1]]
            row[z_ij] = 1
            row[256 + i] = -1
            A.append(row)
            b.append(log_shutter_time[j])

    
    row = [0 for _ in range(256 + num_samples)]
    row[127] = 1 # assume g(127) = 0
    A.append(row)
    b.append(0)

    for i in range(254):
        row = [0 for _ in range(256 + num_samples)]
        row[i] = 1
        row[i + 1] = -2
        row[i + 2] = 1
        A.append(row)
        b.append(0)

    A = np.array(A)
    b = np.array(b)

    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    g = x[:256]
    g_RGB.append(g)

# g_R, g_G, g_B = g_RGB[0], g_RGB[1], g_RGB[2]

#%%
plt.figure(figsize=(16, 9), dpi=100)
y = [i for i in range(256)]

plt.subplot(2, 2, 1)
plt.plot(g_RGB[0], y, 'r')
plt.title('Red')
plt.xlabel('log exposure X')
plt.ylabel('pixel value Z')
plt.xticks(range(-5, 5, 1))
plt.yticks(range(0, 256, 50))

plt.subplot(2, 2, 2)
plt.plot(g_RGB[1], y, 'g')
plt.title('Green')
plt.xlabel('log exposure X')
plt.ylabel('pixel value Z')
plt.xticks(range(-5, 5, 1))
plt.yticks(range(0, 256, 50))

plt.subplot(2, 2, 3)
plt.plot(g_RGB[2], y, 'b')
plt.title('Blue')
plt.xlabel('log exposure X')
plt.ylabel('pixel value Z')
plt.xticks(range(-5, 5, 1))
plt.yticks(range(0, 256, 50))

plt.subplot(2, 2, 4)
plt.plot(g_RGB[0], y, 'r', g_RGB[1], y, 'g', g_RGB[2], y, 'b')
plt.title('Red, Green, and Blue curves')
plt.xlabel('log exposure X')
plt.ylabel('pixel value Z')
plt.xticks(range(-5, 5, 1))
plt.yticks(range(0, 256, 50))

path_response_curves = f'{_img_dir_}/response_curves.jpg'
plt.savefig(path_response_curves)
print(f'Response curves saved at {path_response_curves}')

#%%
####################################
# Construct HDR radiance map
####################################

#%%
def weightFn(z):
    if z <= 127:
        return max(z, 1e-5)
    else:
        return max(255 - z, 1e-5)

E_RGB = []
for channel in range(3):
    E = np.zeros((height, width))
    g = g_RGB[channel]
    for i in range(height):
        for j in range(width):
            weighted_log_E = 0
            weight = 0
            for p in range(num_photos):
                Z_pixel = aligned_RGB_list[channel][p][i][j]
                weighted_log_E += weightFn(Z_pixel) * (g[Z_pixel] - log_shutter_time[p])
                weight += weightFn(Z_pixel)
            E[i][j] = pow(2, weighted_log_E / weight)
    E_RGB.append(E)

R_w, G_w, B_w = E_RGB[0], E_RGB[1], E_RGB[2]
HDR_image = np.dstack((R_w, G_w, B_w))
print(f'HDR_image.shape = {HDR_image.shape}')

#%%
# Show recover result (just for visualization, not real HDR)
plt.figure(figsize=(16, 9), dpi=100)
im = plt.imshow(G_w, cmap='jet', vmin=np.percentile(G_w, 3), vmax=np.percentile(G_w, 97))
cbar = plt.colorbar(im)

path_radiance_visualize = f'{_img_dir_}/radiance_visualize.jpg'
plt.savefig(path_radiance_visualize)
print(f'Radiance visualization saved at {path_radiance_visualize}')

#%%
# Save HDR
path_hdr = f'{_img_dir_}/HDR_image.hdr'
cv2.imwrite(path_hdr, HDR_image.astype(np.float32))
print(f'HDR image saved at {path_hdr}')

#%%
####################################
# Tone mapping
####################################

#%%
def saveTonemapPhotographicGlobal(path, HDR_image, key=0.5, delta=1e-6): # low key = 0.18, high key = 0.5
    L_w = HDR_image
    L_w_bar = pow(2, np.log2(L_w + delta).mean())
    L_m = L_w * key / L_w_bar
    L_white = np.max(L_m)
    L_d = (L_m * (1 + (L_m / pow(L_white, 2)))) / (1 + L_m)

    LDR_image = np.clip(np.array(L_d * 255), 0, 255).astype(np.uint8)
    cv2.imwrite(path, LDR_image)
    return LDR_image

def Lm2Lblur(L_m, s_max=43, phi=8, key=0.5, epsilon=0.05):
    L_blur = np.zeros(L_m.shape)
    s_list = [i for i in range(1, s_max + 1, 2)]
    num_s = len(s_list)
    for channel in range(3):
        L_m_onechannel = L_m[:,:,channel]
        L_s_list = np.zeros(L_m_onechannel.shape + (num_s,))
        V_s_list = np.zeros(L_m_onechannel.shape + (num_s,))
        for i in range(num_s):
            s = s_list[i]
            L_s_list[:,:,i] = cv2.GaussianBlur(L_m_onechannel,(s,s),0)
            V_s_list[:,:,i] = np.absolute((L_s_list[:,:,i] - L_s_list[:,:,max(i-1, 0)]) / (pow(2, phi) * key / pow(s, 2) + L_s_list[:,:,max(i-1, 0)]))
            
        L_s_max = np.argmax(V_s_list > epsilon, axis=2) - 1
        L_s_max[L_s_max < 0] = 0

        for i in range(L_m.shape[0]):
            for j in range(L_m.shape[1]):
                L_blur[i, j, channel] = L_s_list[i, j, L_s_max[i, j]]

    return L_blur

def saveTonemapPhotographicLocal(path, HDR_image, key=0.5, delta=1e-6, s_max=43, phi=8, epsilon=0.05):
    L_w = HDR_image
    L_w_bar = pow(2, np.log2(L_w + delta).mean())
    L_m = L_w * key / L_w_bar
    L_blur = Lm2Lblur(L_m, s_max, phi, key, epsilon)
    L_d = L_m / (1 + L_blur)
    
    LDR_image = np.clip(np.array(L_d * 255), 0, 255).astype(np.uint8)
    cv2.imwrite(path, LDR_image)
    return LDR_image

path_global_map = f'{_img_dir_}/tonemap_photographic_global.png'
path_local_map = f'{_img_dir_}/tonemap_photographic_local.png'
LDR_phtographi_global = saveTonemapPhotographicGlobal(path_global_map, HDR_image)
LDR_phtographi_local = saveTonemapPhotographicLocal(path_local_map, HDR_image)
print(f'Tone mapped images saved at {path_global_map} and {path_local_map}')

#%%
cv2.imwrite('../result.png', LDR_phtographi_local)
print(f'Final result saved at ../result.png')

#%%
