import numpy as np


def random_set_number_matrix(size, num_numbers, number):
    if num_numbers > size**2:
        raise ValueError("num_numbers cannot exceed total pixels")

    # Total number of pixels
    total = size * size

    # Create flat array with the correct number of ones and zeros
    flat = np.zeros(total, dtype=int)
    flat[:num_numbers] = number

    # Shuffle randomly
    np.random.shuffle(flat)

    # Reshape to square matrix
    return flat.reshape((size, size))


def noise_image(image, num_numbers, number):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square")
    noise = random_set_number_matrix(image.shape[0], num_numbers, number).astype(np.uint8)
    image += noise
    image = np.clip(image, 0, 255)
    return image


def pad_image(image, pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0):
    if image.ndim == 2:
        return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                      mode='constant', constant_values=0)
    else:
        raise ValueError("unsupported image shape")


def crop_image(image, crop_top, crop_bottom, crop_left, crop_right):
    h, w = image.shape[:2]
    return image[crop_top:h-crop_bottom, crop_left:w-crop_right]


def uncenter_image(image, top, bottom, left, right):
    padded_image = pad_image(image, top, bottom, left, right)

    top_crop = top + bottom - np.random.randint(0, top + bottom)
    bottom_crop = top + bottom - top_crop

    left_crop = left + right - np.random.randint(0, left + right)
    right_crop = left + right - left_crop

    uncentered_image = crop_image(padded_image, top_crop, bottom_crop, left_crop, right_crop)
    return uncentered_image

