#EG/2019/3633
#Kavinda B.G.K
#Take Home Assignment

import cv2
import numpy as np
import math

def reduce_intensity_levels(image, levels):
    max_val = 255
    factor = max_val / (levels - 1)
    return np.round(image / factor) * factor

def apply_average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def reduce_resolution(image, block_size):
    rows, cols = image.shape[:2]
    for row in range(0, rows, block_size):
        for col in range(0, cols, block_size):
            block = image[row:row+block_size, col:col+block_size]
            avg = np.mean(block)
            image[row:row+block_size, col:col+block_size] = avg
    return image

# Load image
image = cv2.imread('Dog.jpeg')

# Reduce intensity levels
while True:
    levels = int(input("Enter the number of intensity levels (2, 4, 8, 16, ...): "))
    if levels > 0 and math.log2(levels).is_integer():
        break
    else:
        print("Please enter a valid number of intensity levels that is an integer power of 2.")

output_intensity_levels = reduce_intensity_levels(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), levels)

# Apply average filter with different kernel sizes
output_image_3x3 = apply_average_filter(image, 3)
output_image_10x10 = apply_average_filter(image, 10)
output_image_20x20 = apply_average_filter(image, 20)

# Rotate image by 45 and 90 degrees
rotated_45 = rotate_image(image, 45)
rotated_90 = rotate_image(image, 90)

# Reduce resolution with different block sizes.
output_image_3x3_block = reduce_resolution(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).copy(), 3)
output_image_5x5_block = reduce_resolution(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).copy(), 5)
output_image_7x7_block = reduce_resolution(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).copy(), 7)

# Display the results
cv2.imshow('Original Image', image)
# Intensity Levels Image
cv2.imshow('Reduced Intensity Levels Image', output_intensity_levels)
# Average Filter
cv2.imshow('3x3 Average Filter', output_image_3x3)
cv2.imshow('10x10 Average Filter', output_image_10x10)
cv2.imshow('20x20 Average Filter', output_image_20x20)
# Rotate Image
cv2.imshow('Rotated 45 Degrees', rotated_45)
cv2.imshow('Rotated 90 Degrees', rotated_90)
# Block Avg Filter
cv2.imshow('3x3 Block Average', output_image_3x3_block)
cv2.imshow('5x5 Block Average', output_image_5x5_block)
cv2.imshow('7x7 Block Average', output_image_7x7_block)

cv2.waitKey(0)
cv2.destroyAllWindows()

