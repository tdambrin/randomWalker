"""
==========================
Random walker segmentation
==========================

The random walker algorithm [1]_  determines the segmentation of an image from
a set of markers labeling several phases (2 or more). An anisotropic diffusion
equation is solved with tracers initiated at the markers' position. The local
diffusivity coefficient is greater if neighboring pixels have similar values,
so that diffusion is difficult across high gradients. The label of each unknown
pixel is attributed to the label of the known marker that has the highest
probability to be reached first during this diffusion process.

In this example, two phases are clearly visible, but the data are too
noisy to perform the segmentation from the histogram only. We determine
markers of the two phases from the extreme tails of the histogram of gray
values, and use the random walker for the segmentation.

.. [1] *Random walks for image segmentation*, Leo Grady, IEEE Trans. Pattern
       Anal. Mach. Intell. 2006 Nov; 28(11):1768-83 :DOI:`10.1109/TPAMI.2006.233`

"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage
from skimage.color import rgb2gray
import Tkinter
import cv2
from random import randint

CUTCOLOR = (255, 0, 0)

INTTOCOLOR = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (255, 255, 0),
    4: (0, 255, 255),
    5: (100, 28, 200)
}

SOURCE, SINK = -2, -1
#SCREENWIDTH = Tkinter.Tk().winfo_screenmmwidth()
SF = 10 #scale factor
LOADSEEDS = False


def plantSeed(image):
    def drawLines(x, y, seedN, seedColor):
        #color = INTTOCOLOR[seedN]
        code = seedN
        cv2.circle(image, (x, y), radius, INTTOCOLOR[code], thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)
        # seeds[y//SF][x//SF] = code

    def onMouse(event, x, y, flags, params):
        global drawing
        seedN = params[0]
        seedColor = params[1]
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, seedN, seedColor)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, seedN, seedColor)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(seedingN):
        rgb = np.random.random_integers(0, 256, 3)
        rgb[2] = 0
        alldone = False
        global drawing
        drawing = False
        windowname = "Planting seeds, enter : next label, del : previous label"
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(windowname, onMouse, (seedingN, rgb))
        while (1):
            cv2.imshow(windowname, image)
            pressed = cv2.waitKey(33) & 0xFF
            if pressed == 27:
                alldone = True
                break
            elif pressed == 13:
                seedingN += 1
                break
            elif pressed == 8:
                if seedingN > 1:
                    seedingN -= 1
                    break
        cv2.destroyAllWindows()
        return alldone, seedingN

    seeds = np.zeros(image.shape, dtype='uint8')
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_GRAY2RGB)
    print('converted to gray with cv2')
    image = cv2.resize(image, (0, 0), fx=SF // 10, fy=SF // 10)
    print('resized with cv2', image.shape)
    radius = image.shape[0] // 50
    thickness = -1  # fill the whole circle
    global drawing
    drawing = False
    seedingNum = 1
    plantRes = paintSeeds(seedingNum)
    while (not plantRes[0]):
        plantRes = paintSeeds(plantRes[1])

    '''seeds = np.zeros(image.shape, dtype=np.uint)
    seeds[image > 0.8] = 1
    seeds[image < 0.3] = 2'''
    # print(seeds)
    return seeds, image


# toIci

def containsOnes(tab):
    for elem in tab:
        if 1 in elem:
            return True
    return False


def generateInput(img):
    # Generate noisy synthetic data
    # data = skimage.img_as_float(binary_blobs(length=128, seed=1))
    data = rgb2gray(img)
    data = skimage.img_as_float(data)
    print(data[0][0])
    # data = cv2.resize(data, dsize=(20,20))
    # data = rescale_intensity(data, in_range=(0.25,0.9), out_range=(-1, 1))
    print('RANGE : ', imgMax(data), imgMin(data))

    '''
    data = skimage.img_as_float(binary_blobs(length=128, seed=1))
    sigma = 0.35
    data += np.random.normal(loc=0, scale=sigma, size=data.shape)
    data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                             out_range=(-1, 1))'''
    '''
    # The range of the binary image spans over (-1, 1).
    # We choose the hottest and the coldest pixels as markers.
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < -0.95] = 1
    markers[data > 0.95] = 2'''

    # return data, markers
    markers, seeded = plantSeed(data)
    return data, markers


def introduceNoise(img, sigma):
    res = img
    res += np.random.normal(loc=0, scale=sigma, size=res.shape)
    return res


def imgMax(image):
    res = image[0].max()
    for i in range(image.shape[0]):
        if image[i].max() > res:
            res = image[i].max()
    return res


def imgMin(image):
    res = image[0].min()
    for i in range(image.shape[0]):
        if image[i].min() > res:
            res = image[i].min()
    return res


def plot_res(data, markers, labels):
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                        sharex=True, sharey=True)
    ax1.imshow(data, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Noisy data')
    ax2.imshow(markers, cmap='magma')
    ax2.axis('off')
    ax2.set_title('Markers')
    ax3.imshow(labels, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Segmentation')

    fig.tight_layout()
    plt.savefig('results.png', format='png')
    plt.show()

def getDivisionIndexes(labels):
    neighborsRelativeIndexes = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    res = []
    for i, raw in enumerate(labels):
        for j, cell in enumerate(raw):
            assigned = cell
            try:
                for k in range(4):
                    if labels[i + neighborsRelativeIndexes[k][0]][j + neighborsRelativeIndexes[k][1]] != cell:
                        res.append((i, j))
            except IndexError:
                pass
    return res


def toPlot(initData, markers, labels):
    segmentedImg = cv2.cvtColor(initData.astype('float32'), cv2.COLOR_GRAY2RGB)
    plot_markers = cv2.cvtColor(initData.astype('float32'), cv2.COLOR_GRAY2RGB)
    divIndexes = getDivisionIndexes(labels)
    for ind in divIndexes:
        #segmentedImg[ind[0]][ind[1]] = np.array([255, 0, 0])
        segmentedImg[ind[0]][ind[1]] = CUTCOLOR
    for i, raw in enumerate(markers):
        for j, cell in enumerate(raw):
            if cell != 0:
                plot_markers[i][j] = INTTOCOLOR[cell]
    return segmentedImg, plot_markers


if __name__ == '__main__':
    # Run random walker algorithm
    initData = rgb2gray(skimage.data.astronaut())
    #initData = introduceNoise(initData, 0.1)
    data, markers = generateInput(initData)
    labels = random_walker(data, markers, mode='bf')
    segmented, plMarkers = toPlot(initData, markers, labels)
    plot_res(data, plMarkers, segmented)
