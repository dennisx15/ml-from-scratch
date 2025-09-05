"""
=====================================================
 Image Manipulation Utilities
=====================================================

This module provides functions for creating, modifying, and augmenting
square images represented as NumPy arrays.

Functions include:
 - random_set_number_matrix: generate a square matrix with a specified
   number of repeated values randomly placed
 - noise_image: add random noise to an image
 - pad_image: pad an image with zeros on any side
 - crop_image: crop edges of an image
 - uncenter_image: randomly shift the image within a padded frame

Author: Dennis Alacahanli
Purpose: Educational project to understand data augmentation and how it affects training
"""

import numpy as np


# ----------------- Random Matrix with Specific Values ----------------- #
def random_set_number_matrix(size, num_numbers, number):
    """
    Generate a square matrix of given size with `num_numbers` randomly placed
    occurrences of `number`. Remaining pixels are zeros.

    Parameters:
        size (int): Width and height of the square matrix
        num_numbers (int): Number of occurrences of `number` to place
        number (int): The number to place in the matrix

    Returns:
        np.ndarray: Square matrix of shape (size, size)
    """
    if num_numbers > size ** 2:
        raise ValueError("num_numbers cannot exceed total pixels")

    total = size * size  # Total number of pixels

    # Create flat array with specified number of `number`s and zeros
    flat = np.zeros(total, dtype=int)
    flat[:num_numbers] = number

    # Shuffle randomly
    np.random.shuffle(flat)

    # Reshape flat array into square matrix
    return flat.reshape((size, size))


# ----------------- Add Random Noise to an Image ----------------- #
def noise_image(image, num_numbers, number):
    """
    Add random noise to a square image by adding a matrix of randomly placed
    values.

    Parameters:
        image (np.ndarray): Square image array
        num_numbers (int): Number of noise values to add
        number (int): Noise value to add

    Returns:
        np.ndarray: Image with noise added
    """
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square")

    noise = random_set_number_matrix(image.shape[0], num_numbers, number).astype(np.uint8)
    image += noise
    image = np.clip(image, 0, 255)  # Ensure pixel values remain valid
    return image


# ----------------- Pad Image with Zeros ----------------- #
def pad_image(image, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):
    """
    Pad a 2D image with zeros on specified sides.

    Parameters:
        image (np.ndarray): 2D image
        pad_top, pad_bottom, pad_left, pad_right (int): Number of pixels to pad

    Returns:
        np.ndarray: Padded image
    """
    if image.ndim == 2:
        return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                      mode='constant', constant_values=0)
    else:
        raise ValueError("unsupported image shape")


# ----------------- Crop Image ----------------- #
def crop_image(image, crop_top, crop_bottom, crop_left, crop_right):
    """
    Crop pixels from the edges of an image.

    Parameters:
        image (np.ndarray): Input image
        crop_top, crop_bottom, crop_left, crop_right (int): Pixels to remove

    Returns:
        np.ndarray: Cropped image
    """
    h, w = image.shape[:2]
    return image[crop_top:h - crop_bottom, crop_left:w - crop_right]


# ----------------- Randomly Uncenter Image ----------------- #
def uncenter_image(image, top, bottom, left, right):
    """
    Randomly shift an image within a padded frame, then crop to original size.

    Parameters:
        image (np.ndarray): Input image
        top, bottom, left, right (int): Maximum padding on each side

    Returns:
        np.ndarray: Randomly uncentered image
    """
    # Pad the image with zeros
    padded_image = pad_image(image, top, bottom, left, right)

    # Randomly determine how many pixels to crop from each side
    top_crop = top + bottom - np.random.randint(0, top + bottom)
    bottom_crop = top + bottom - top_crop
    left_crop = left + right - np.random.randint(0, left + right)
    right_crop = left + right - left_crop

    # Crop to original image size
    uncentered_image = crop_image(padded_image, top_crop, bottom_crop, left_crop, right_crop)
    return uncentered_image
