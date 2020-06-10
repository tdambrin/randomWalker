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
#import Tkinter
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
        color = INTTOCOLOR[seedN]
        code = seedN
        cv2.circle(localImg, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x, y), radius, code, thickness)
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
        rgb[1] = 0
        rgb[2] = 0
        alldone = False
        global drawing
        drawing = False
        windowname = "Planting seeds, enter : next label, del : previous label"
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(windowname, onMouse, (seedingN, rgb))
        while (1):
            cv2.imshow(windowname, localImg)
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

    localImg = image.copy()
    initially_gray = len(localImg.shape) < 3
    if initially_gray:
        localImg = cv2.cvtColor(localImg.astype('float32'), cv2.COLOR_GRAY2RGB)
    #image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    localImg = cv2.resize(localImg, (512, 512))
    localImg = localImg.astype('float32')
    returnImg = localImg.copy()
    if initially_gray:
        returnImg = rgb2gray(returnImg)
    seeds = np.zeros(returnImg.shape, dtype='uint8')
    radius = localImg.shape[0] // 50
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
    return seeds, returnImg

def containsOnes(tab):
    for elem in tab:
        if 1 in elem:
            return True
    return False

def compare(inf):
    if inf:
        return lambda a, b: a < b
    else:
        return lambda a, b: a > b


def mark(img, markers, label, threshold, inf):
    cmp = compare(inf)
    for i, raw in enumerate(img):
        for j, cell in enumerate(raw):
            if cmp(cell, threshold):
                markers[i][j] = label
    return markers

def generateInput(img, autoseed):
    # Generate noisy synthetic data
    # data = skimage.img_as_float(binary_blobs(length=128, seed=1))
    #data = rgb2gray(img)
    data = img.copy()
    #data = skimage.img_as_float(data)
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
    if not autoseed:
        markers, formated = plantSeed(data)
    else:
        formated = img.copy()
        formated = cv2.resize(formated.astype('float32'), (512, 512))
        markers = np.zeros(formated.shape, dtype='uint8')
        minval = imgMin(img)
        maxval = imgMax(img)
        markers = mark(formated, markers, 1, ((maxval - minval) / 10), True)
        markers = mark(formated, markers, 3, maxval - ((maxval - minval) / 5), False)

    return formated, markers


def introduceNoise(img, sigma, color):
    res = img.copy()
    if not color:
        res = np.add(np.random.normal(loc=0, scale=sigma, size=res.shape), res, casting='unsafe')
    else:
        res = np.add(np.random.normal(loc=0, scale=sigma, size=res.shape), res, casting='unsafe')
    return res


def imgMax(image):
    if len(image.shape) == 3:
        res = image[0][0]
        for i, raw in enumerate(image):
            for j, cell in enumerate(raw):
                if sum(cell) > sum(res):
                    res = cell

    else:
        res = image[0].max()
        for i, raw in enumerate(image):
            if raw.max() > res:
                res = raw.max()
    return res


def imgMin(image):
    if len(image.shape) == 3:
        res = image[0][0]
        for i, raw in enumerate(image):
            for j, cell in enumerate(raw):
                if sum(cell) < sum(res):
                    res = cell

    else:
        res = image[0].min()
        for i, raw in enumerate(image):
            if raw.min() < res:
                res = raw.min()
    return res


def plot_res(data, markers, labels, algo):
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
    plt.savefig('results'+algo+'.png', format='png')
    plt.show()

def getDivisionIndexes(labels):
    neighborsRelativeIndexes = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    res = []
    threeD = len(labels.shape) == 3
    for i, raw in enumerate(labels):
        for j, cell in enumerate(raw):
            assigned = cell
            try:
                for k in range(4):
                    if not threeD:
                        if labels[i + neighborsRelativeIndexes[k][0]][j + neighborsRelativeIndexes[k][1]] != cell:
                            res.append((i, j))
                    else:
                        if labels[i + neighborsRelativeIndexes[k][0]][j + neighborsRelativeIndexes[k][1]][0] != cell[0]:
                            res.append((i, j))
            except IndexError:
                pass
    return res


def toPlot(initData, markers, labels):
    if len(initData.shape) < 3:
        segmentedImg = cv2.cvtColor(initData.astype('float32'), cv2.COLOR_GRAY2RGB)
        plot_markers = cv2.cvtColor(initData.astype('float32'), cv2.COLOR_GRAY2RGB)
    else:
        segmentedImg = initData.copy()
        plot_markers = initData.copy()
    #print('LABELS', labels)
    divIndexes = getDivisionIndexes(labels)
    for ind in divIndexes:
        #segmentedImg[ind[0]][ind[1]] = np.array([255, 0, 0])
        segmentedImg[ind[0]][ind[1]] = CUTCOLOR
    for i, raw in enumerate(markers):
        for j, cell in enumerate(raw):
            if type(cell) == np.uint8 and cell != 0:
                plot_markers[i][j] = INTTOCOLOR[cell]
            elif type(cell) != np.uint8 and cell[0] != 0:
                plot_markers[i][j] = INTTOCOLOR[cell[0]]
    return segmentedImg, plot_markers


if __name__ == '__main__':
    # Run random walker algorithm
    initData = rgb2gray(skimage.data.astronaut())
    #initData = introduceNoise(initData, 0.1)
    data, markers = generateInput(initData)
    labels = random_walker(data, markers, mode='cg_mg')
    segmented, plMarkers = toPlot(initData, markers, labels)
    plot_res(data, plMarkers, segmented)
