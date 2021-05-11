"""
Comparing grayscale histogram equalization and CLAHE
"""

# Import required packages:
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(2, 3, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


# Create the dimensions of the figure and set title:
plt.figure(figsize=(14, 8))
plt.suptitle("Grayscale histogram equalization with cv2.calcHist() and CLAHE", fontsize=16, fontweight='bold')

# Load the image and convert it to grayscale:
image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the histogram calling cv2.calcHist()
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Equalize image and calculate histogram
gray_image_eq = cv2.equalizeHist(gray_image)
hist_eq = cv2.calcHist([gray_image_eq], [0], None, [256], [0, 256])

# Create clahe:
clahe = cv2.createCLAHE(clipLimit=4.0)

# Apply CLAHE to the gryscale image and calculate histogram:
gray_image_clahe = clahe.apply(gray_image)
hist_clahe = cv2.calcHist([gray_image_clahe], [0], None, [256], [0, 256])

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 4, 'm')
show_img_with_matplotlib(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 2)
show_hist_with_matplotlib_gray(hist_eq, "grayscale equalized histogram", 5, 'm')
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 3)
show_hist_with_matplotlib_gray(hist_clahe, "grayscale clahe histogram", 6, 'm')

# Show the Figure:
plt.show()
