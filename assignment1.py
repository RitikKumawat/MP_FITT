import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import skimage as io
import cv2
from google.colab.patches import cv2_imshow
     

image1 = cv2.imread("/content/drive/MyDrive/3d4288eb-d58b-48ea-958f-a3006d2fc2f1.jpg")
image1 = np.array(image1)
gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

image2 = cv2.imread("/content/drive/MyDrive/5fc8aedf-ec10-4c6c-ae0d-1ebee868ed18.jpg")
image2 = np.array(image2)
gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
     

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout()
ax[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
ax[0].set_title('Image')

ax[1].imshow(gray1, cmap='gray')  # Set colormap to grayscale
ax[1].set_title('Gray Image')
plt.show()
     


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout()
ax[0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
ax[0].set_title('Image')

ax[1].imshow(gray2, cmap='gray')  # Set colormap to grayscale
ax[1].set_title('Gray Image')
plt.show()
     

Local Binary Pattern Function


def Binary_pattern(img, x, y):
    center = img[x][y]
    values = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            values.append(1 if img[x + i][y + j] >= center else 0)
    val = 0
    for i in range(len(values)):
      val+=values[i]*2**i

    return val
     

def function_1(img):
    height, width = img.shape
    lbp_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            lbp_img[i, j] = Binary_pattern(img, i, j)
    return lbp_img
     

Binary_pattern_image_1 = function_1(gray1)
Binary_pattern_image_2 = function_1(gray2)
     

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout()
ax[0].imshow(Binary_pattern_image_1, cmap='gray')
ax[0].set_title('Binary_pattern_image_1')
ax[1].imshow(Binary_pattern_image_2, cmap='gray')
ax[1].set_title('Binary_pattern_image_2')
plt.show()
