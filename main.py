#%% require package
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% some parameters
in_path = './input.png'
out_path = './output.png'

#%% load image
in_im = cv2.imread(in_path)
HEIGHT, WIDTH, CHANNEL = in_im.shape
plt.imshow(in_im)
plt.show()


# %% get contour
in_im_gray = cv2.cvtColor(in_im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(in_im_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# %% create 
# set to white but not transparent
blank_arr = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
blank_arr[:, :] = [255, 255, 255, 0]
plt.imshow(blank_arr)

# %% fill
cv2.fillPoly(blank_arr, pts=contours, color=(255, 255, 255, 255))
cv2.imwrite(out_path, blank_arr)