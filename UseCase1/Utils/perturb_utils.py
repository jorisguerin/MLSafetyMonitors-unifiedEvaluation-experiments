import cv2
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

from Params.params_perturb import *


# Sensor failure perturbations
def shifted_pixels(image, max_shift):
    """
    Randomly shifts pixels locations within a distance of max_shift.

    Args:
        image (np.array): Original image
        max_shift (int): Max distance

    Returns:
        perturbed image
    """
    copy = np.copy(image)

    m, n = copy.shape[0], copy.shape[1]
    col_start = np.random.randint(0, max_shift, copy.shape[0])
    idx = np.mod(col_start[:, None] + np.arange(n), n)
    copy = copy[np.arange(m)[:, None], idx]

    return copy


def pixel_trap(image, n_rows):
    """
    Randomly deletes n_rows rows of the original image.

    Args:
        image (np.array): Original image
        n_rows (int): Number of rows to remove

    Returns:
        perturbed image
    """
    copy = np.copy(image)

    indices = np.random.choice(copy.shape[0], n_rows, replace=False)
    copy[indices] = 0

    return copy


# External perturbations
def brightness(image, intensity):
    """
    Modifies brightness of the image.

    Args:
        image (np.array): Original image
        intensity (float): brightness intensity (0.5 is already really bright)

    Returns:
        perturbed image
    """
    copy = np.copy(image)

    copy = rgb2hsv(copy)
    copy[:, :, 2] = np.clip(copy[:, :, 2] + intensity, 0, 1)
    copy = hsv2rgb(copy)
    copy = (copy * 255).astype(np.uint8)
    return copy


def _plasma_fractal(mapsize=4096, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(image, intensity):
    """
    Generates fog on the image.
    Args:
        image (np.array): Original image
        intensity (int): Index from the severity level defined in params_perturb

    Returns:
        Perturbed image
    """
    intensity = fog_intensity_lvls[intensity]
    copy = np.copy(image)

    dims = copy.shape
    copy = cv2.copyMakeBorder(image,
                              (fractal_mapsize - dims[0]) // 2,
                              (fractal_mapsize - dims[0]) // 2,
                              (fractal_mapsize - dims[1]) // 2,
                              (fractal_mapsize - dims[1]) // 2,
                              cv2.BORDER_CONSTANT, 0)

    copy = np.array(copy) / 255.
    max_val = copy.max()
    copy += intensity[0] * _plasma_fractal(fractal_mapsize,
                                           intensity[1])[:, :][..., np.newaxis]
    copy = np.clip(copy * max_val / (max_val + copy[0]), 0, 1) * 255
    copy = copy[(fractal_mapsize - dims[0]) // 2:fractal_mapsize-(fractal_mapsize - dims[0]) // 2,
                (fractal_mapsize - dims[1]) // 2:fractal_mapsize-(fractal_mapsize - dims[1]) // 2]
    copy = copy.astype(np.uint8)

    return copy


# Movement perturbation
def motion_blur(image, intensity, direction="vertical"):
    """
    Generates a motion blur on the image
    Args:
        image (np.array): Original image
        intensity (int): Kernel size, higher implies blurrier image
        direction (str): "vertical" (default) or "horizontal"

    Returns:
        perturbed image
    """

    if direction not in ["vertical", "horizontal"]:
        print("Invalid value for 'direction', please select either 'horizontal'or 'vertical'")
        return

    kernel_size = intensity
    kernel = np.zeros((kernel_size, kernel_size))
    if direction == "vertical":
        kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    else:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    output = cv2.filter2D(image, -1, kernel)

    return output
