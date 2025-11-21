# Image Processing Lab Codes (Python + OpenCV)
## **1. Greyscale to Binary Conversion**

import cv2
img = cv2.imread('input.jpg', 0)
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite('binary.jpg', binary)

## **2. RGB to Greyscale Conversion**

import cv2
img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', gray)

## **3. Add Border to an Image**

import cv2
img = cv2.imread('input.jpg')
border = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0,0,0])
cv2.imwrite('bordered.jpg', border)

## **4. Complement of an Image**

import cv2
img = cv2.imread('input.jpg')
complement = 255 - img
cv2.imwrite('complement.jpg', complement)

## **5. Log Transform**

import cv2
import numpy as np
img = cv2.imread('input.jpg', 0)
c = 255 / np.log(1 + np.max(img))
log_img = c * (np.log(img + 1))
log_img = np.array(log_img, dtype=np.uint8)
cv2.imwrite('log.jpg', log_img)

## **6. Power Law Transform (Gamma Correction)**

import cv2
import numpy as np
img = cv2.imread('input.jpg', 0)
gamma = 2.2
norm = img / 255.0
power = np.power(norm, gamma)
out = np.uint8(power * 255)
cv2.imwrite('power.jpg', out)

## **7. Contrast Stretching**

import cv2
import numpy as np
img = cv2.imread('input.jpg', 0)
min_val = np.min(img)
max_val = np.max(img)
stretched = (img - min_val) * (255 / (max_val - min_val))
stretched = np.uint8(stretched)
cv2.imwrite('contrast_stretch.jpg', stretched)

## **8. Histogram Equalization**

import cv2
img = cv2.imread('input.jpg', 0)
eq = cv2.equalizeHist(img)
cv2.imwrite('hist_eq.jpg', eq)

## **9. Histogram Matching**

import cv2
import numpy as np
from skimage.exposure import match_histograms

src = cv2.imread('source.jpg', 0)
ref = cv2.imread('reference.jpg', 0)
matched = match_histograms(src, ref)
cv2.imwrite('matched.jpg', matched)

## **10. Image Smoothening (Blurring)**

import cv2
img = cv2.imread('input.jpg')
blur = cv2.GaussianBlur(img, (5,5), 0)
cv2.imwrite('smooth.jpg', blur)

## **11. Bit Plane Slicing**

import cv2
import numpy as np
img = cv2.imread('input.jpg', 0)
bits = []
for i in range(8):
    bit = (img >> i) & 1
    bit_img = bit * 255
    cv2.imwrite(f'bitplane_{i}.jpg', bit_img)

## **12. Image Sharpening**

import cv2
import numpy as np
img = cv2.imread('input.jpg', 0)
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharp = cv2.filter2D(img, -1, kernel)
cv2.imwrite('sharpen.jpg', sharp)

## **13. Implement CNN (Simple Classification Using Keras)**

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])