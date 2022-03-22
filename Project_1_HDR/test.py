#%%
import cv2
import numpy as np
hdr = cv2.imread('./data/HDR_image.hdr', flags=cv2.IMREAD_ANYDEPTH)
tonemap = cv2.createTonemap(2.8)
ldr = tonemap.process(hdr) * 255
cv2.imwrite('./test.png', ldr)
#%%
