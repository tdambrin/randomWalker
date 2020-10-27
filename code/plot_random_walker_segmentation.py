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
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker, watershed
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage
from skimage.color import rgb2gray
import skimage.filters
#import Tkinter
import cv2
import math
from random import randint
from matplotlib.widgets import Button

CUTCOLOR = (255, 0, 0)

INTTOCOLOR = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (255, 255, 0),
    4: (0, 255, 255),
    5: (100, 28, 200)
}
#SCREENWIDTH = Tkinter.Tk().winfo_screenmmwidth()

RADIUS = 10

NAMESTODATA = {'astronaut': skimage.data.astronaut(),
               'coins': skimage.data.coins(),
               'immuno': skimage.data.immunohistochemistry()}

class AddedToAlreadyExisting(Exception):
    """ class used when autoseeding from local intensities, indicate that a newly found center is added to set of pixels
    corresponding to an already found label """
    pass

def plantSeed(image):
    def drawLines(x, y, seedN, seedColor):
        color = INTTOCOLOR.get(seedN, (255, 255, 255))
        code = seedN
        if RADIUS > 0:
            cv2.circle(localImg, (x, y), RADIUS, color, thickness)
            cv2.circle(seeds, (x, y), RADIUS, code, thickness)
        # localImg[y][x] = color
        # seeds[y][x] = code
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
        global RADIUS
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
            elif pressed == 43:
                RADIUS += (RADIUS + 5) // 5 #20 percent increase
                print(RADIUS)
            elif pressed == 45:
                RADIUS -= (RADIUS + 5) // 5 #20 percent decrease
                print(RADIUS)

        cv2.destroyAllWindows()
        return alldone, seedingN

    localImg = image.copy()
    initially_gray = len(localImg.shape) < 3
    if initially_gray:
        localImg = cv2.cvtColor(localImg.astype('float32'), cv2.COLOR_GRAY2RGB)
    #image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    #localImg = cv2.resize(localImg, (512, 512))
    localImg = localImg.astype('float32')
    returnImg = localImg.copy()
    if initially_gray:
        returnImg = rgb2gray(returnImg)
    seeds = np.zeros(returnImg.shape, dtype=np.uint8)
    global RADIUS
    RADIUS = localImg.shape[0] // 50
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


def markFromThreshold(img, markers, label, threshold, inf):
    print(threshold)
    cmp = compare(inf)
    if len(img.shape) < 3:
        for i, raw in enumerate(img):
            for j, cell in enumerate(raw):
                if cmp(cell, threshold):
                    markers[i][j] = label
    else:
        for i, raw in enumerate(img):
            for j, cell in enumerate(raw):
                if cmp(sum(cell),threshold):
                    markers[i][j][0] = label
    return markers

def markFromLocalIntensity(img, markers, regionNumber):
    print('min : {}'.format(imgMin(img)))
    print('max : {}'.format(imgMax(img)))
    indexAndIntensity = {}
    regionAdded = 0
    #get the center of 10x10 regions with similar values
    for i, raw in enumerate(img):
        for j, cell in enumerate(raw):
            localSimilar = True #True if neighoring pixels have a close intensity
            try:
                for h in range(-5, 5):
                    for v in range(-5, 5):
                        if abs(cell - img[i + h][j + v]) > 0.2:
                            raise IndexError
            except IndexError:
                localSimilar = False

            if localSimilar:
                try:
                    for _, alreadyIn in enumerate(indexAndIntensity.keys()):
                        if abs(alreadyIn - cell) < 0.1: #the newly found center may correspond to an already detected intensity
                            indexAndIntensity[alreadyIn].append((i, j))
                            raise AddedToAlreadyExisting
                    if regionAdded < regionNumber:
                        indexAndIntensity[cell] = []
                        indexAndIntensity[cell].append((i, j))
                        regionAdded += 1
                    else:
                        '''closer = indexAndIntensity.keys()[0]
                        for _, intensity in enumerate(indexAndIntensity.keys()):
                            if abs(cell - intensity) < abs(cell - closer):
                                closer = intensity
                        indexAndIntensity[closer].append((i, j))'''
                except AddedToAlreadyExisting:
                    pass

    print('INDEXINTE len : ', len(indexAndIntensity.keys()), len(indexAndIntensity[indexAndIntensity.keys()[0]]))
    print(indexAndIntensity.keys())
    #approcimate the number of labels based on the following hypothesis : one range of intensity values => one label
    for labelNumber, intensity in enumerate(indexAndIntensity.keys()):
        for _, center in enumerate(indexAndIntensity[intensity]):
            markers[center[0]][center[1]] = labelNumber + 1
        print('New labdel number = ', labelNumber + 1)
    return markers

def markFromHisto(markers, image, peaks):
    for code, onePeak in enumerate(peaks):
        markers[abs(image - onePeak) < 0.1] = code+1
    return markers

def markFromHistoAndTopo(img, markers, peaks): # very inefficient at last
    added = {}
    closeRadius = min(img.shape[:2]) // 3 # max distance with which we will mark two pixels belonging to same region
    minDist = closeRadius // 2 #min distance from boards to be marked
    print('CLOSE RAD = {}'.format(closeRadius))
    for onePeak in peaks:
        added[onePeak] = []
    for i, raw in enumerate(img):
        for j, pix in enumerate(raw):
            try:
                for code, onePeak in enumerate(peaks):
                    if abs(pix - onePeak) < 0.1:
                        if len(added[onePeak]) > 0:
                            for oneAdded in added[onePeak]:
                                if math.hypot(i - oneAdded[0], j - oneAdded[1]) < closeRadius:
                                    if minDist < i < (img.shape[0] - minDist):
                                        if minDist < j < img.shape[1] - minDist:
                                            markers[i][j] = code + 1
                                            added[onePeak].append((i,j))
                                            raise AddedToAlreadyExisting
                        elif minDist < i < (img.shape[0] - minDist) :
                            if minDist < j < img.shape[1] - minDist:
                                markers[i][j] = code + 1
                                added[onePeak].append((i, j))
            except AddedToAlreadyExisting:
                pass

    return markers

def MarkFromMultiOtsu(img, markers, nRegions):
    thresholds = skimage.filters.threshold_multiotsu(img, classes=nRegions)
    for i, thresh in enumerate(thresholds):
        if i == 0:
            markers[img < imgMin(img) + 0.5*thresh] = 1
        elif i == nRegions -1:
            markers[img<imgMin(img) - 0.5*thresh] = nRegions
        else:
            markers[abs(img - thresh) < 0.2] = i + 1
    return markers

def markFromIsoData(img, markers):
    if len(img.shape) > 2:
        raise ValueError('CANNOT USE ISODATA WITH COLORFUL IMAGES')
    seuils = skimage.filters.threshold_isodata(img, return_all=True)
    print('SEUILS :',seuils)
    markers[img < (imgMin(img) + 0.3*seuils[0])] = 1
    markers[img > (imgMax(img) - 0.3*seuils[0])] = 2
    return markers


def generateInput(img, autoseed):
    # Generate noisy synthetic data
    # data = skimage.img_as_float(binary_blobs(length=128, seed=1))
    #data = rgb2gray(img)
    data = img.copy()
    data = skimage.img_as_float(data)
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
        localI = True
        fromThreshold = False
        histAndTopo = False # dont set to true
        histoOnly = True
        multiOtsu = False
        isodata = False

        formated = img.copy()
        #formated = cv2.resize(formated.astype('float32'), (1024, 1024))
        formated = formated.astype('float32')
        markers = np.zeros(formated.shape, dtype='uint8')
        print('Autoseeding from local i')
        if localI:
            markers = markFromLocalIntensity(formated, markers, 2)
            print('Autoseeding over')

        elif fromThreshold:
            minval = imgMin(img)
            maxval = imgMax(img)
            markers = markFromThreshold(formated, markers, 1, minval + ((maxval - minval) / 10), True)
            markers = markFromThreshold(formated, markers, 2, maxval - ((maxval - minval) / 10), False)
        else: # from histogram
            objectsN = 2 #approximated number of objects in the image
            hist, bins_center = skimage.exposure.histogram(formated)
            peaks = [bins_center[i] for i in range(len(hist)) if hist[i] > 10000]
            print('PEAKS : {}'.format(peaks))
            '''if len(peaks) == 0:
                maxOccurence = hist.max()
                peaks = [bins_center[i] for i in range(len(hist)) if hist[i] == maxOccurence]'''
            peaksN = len(peaks)
            while peaksN < objectsN:
                notInMaxOcc = sorted(hist)[-(peaksN + 1)]
                toAppend = [bins_center[i] for i in range(len(hist)) if hist[i] == notInMaxOcc]
                peaks.append(toAppend[0])
                peaksN += 1
            if histAndTopo:
                markers = markFromHistoAndTopo(formated, np.zeros(formated.shape, dtype=np.uint8), peaks)
            elif histoOnly:
                markers = markFromHisto(np.zeros(formated.shape, dtype=np.uint8), formated, peaks)
            elif multiOtsu:
                markers = MarkFromMultiOtsu(img,markers, 3)
            elif isodata:
                markers = markFromIsoData(formated, np.zeros(formated.shape, dtype=np.uint8))
            else:
                raise ValueError('MUST SELECT AN AUTO MARKING METHOD')

    '''print(hist)
    print(bins_center)
    plt.figure(figsize=(9, 4))
    plt.subplot(133)
    plt.plot(bins_center, hist, lw=2)
    plt.tight_layout()
    plt.show()'''

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
        res = sum(image[0][0])
        for i, raw in enumerate(image):
            for j, cell in enumerate(raw):
                if sum(cell) > res:
                    res = sum(cell)
    else:
        res = image[0].max()
        for i, raw in enumerate(image):
            if raw.max() > res:
                res = raw.max()
    return res


def imgMin(image):
    if len(image.shape) == 3:
        res = sum(image[0][0])
        for i, raw in enumerate(image):
            for j, cell in enumerate(raw):
                if sum(cell) < res:
                    res = sum(cell)
    else:
        res = image[0].min()
        for i, raw in enumerate(image):
            if raw.min() < res:
                res = raw.min()
    return res

def plot_res(data, fileName, markers, segmented, labels, instances, algo, segparam):
    def recomputeInc(event):
        global newsegparam
        newsegparam += incordec[algo]
        plt.close(fig)
        plt.close(btns)

    def recomputeDec(event):
        global newsegparam
        newsegparam -= incordec[algo]
        plt.close(fig)
        plt.close(btns)

    def closeFigs(event):
        plt.close(fig)
        plt.close(btns)

    global newsegparam
    newsegparam = segparam
    incordec = { #random walker's and watershed's params doesnt have the same sensibility
        'RW': 100,
        'WD': 0.01
    }
    # Plot results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4), sharex=True, sharey=True)
    ax1.imshow(data, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Input data')
    ax2.imshow(markers, cmap='magma')
    ax2.axis('off')
    ax2.set_title('Markers')
    ax3.imshow(segmented, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Segmentation, Param = {}'.format(segparam))
    ax4.imshow(instances)
    ax4.axis('off')
    ax4.set_title('Classes')
    fig.tight_layout()

    # Saving results
    path = os.getcwd()
    path = path.split('random')[0]
    path += '/images/segmented/'
    plt.savefig(path + 'results' + fileName + algo+'.png', format='png')
    cv2.imwrite(path + 'mask' + fileName + algo + '.png', instances)
    cv2.imwrite(path + 'segment' + fileName + algo + '.png', segmented)
    cv2.imwrite(path + 'labels' + fileName + algo + '.png', labels)

    btns, (btn1, btn2, btn3) = plt.subplots(3, 1, figsize=(2, 5), facecolor='#c0d6e4')
    bredoInc = Button(ax=btn1,
                      label='RedoInc',
                      color='#b96d56',
                      hovercolor='#b96dff')
    bredoInc.color = 'teal'
    bredoInc.on_clicked(recomputeInc)

    bredoDec = Button(ax=btn2,
                      label='RedoDec',
                      color='#b96d56',
                      hovercolor='#b96dff')
    bredoDec.on_clicked(recomputeDec)

    bclose = Button(ax=btn3,
                    label='Close',
                    color='#b96d56',
                    hovercolor='#b96dff')
    bclose.on_clicked(closeFigs)

    plt.show()
    return newsegparam

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
    colored = True
    if len(initData.shape) < 3:
        colored = False
        segmentedImg = cv2.cvtColor(initData.astype('float32'), cv2.COLOR_GRAY2RGB)
        instances = cv2.cvtColor(initData.astype('float32'), cv2.COLOR_GRAY2RGB)
        plot_markers = cv2.cvtColor(initData.astype('float32'), cv2.COLOR_GRAY2RGB)
    else:
        segmentedImg = initData.copy()
        instances = initData.copy()
        plot_markers = initData.copy()
    #print('LABELS', labels)
    divIndexes = getDivisionIndexes(labels)
    print(segmentedImg.shape, instances.shape, plot_markers.shape, initData.shape, markers.shape)
    for ind in divIndexes:
            segmentedImg[ind[0]][ind[1]] = CUTCOLOR
    for i, raw in enumerate(markers):
        for j, cell in enumerate(raw):
            if not colored:
                instances[i][j] = INTTOCOLOR.get(labels[i][j], (0, 0, 0))
                if cell != 0:
                    plot_markers[i][j] = INTTOCOLOR.get(cell, (255, 255, 255))
            else:
                try:
                    instances[i][j] = INTTOCOLOR.get(labels[i][j][0], (0, 0, 0))
                except IndexError:
                    pass
                if cell[0] != 0:
                    try:
                        plot_markers[i][j] = INTTOCOLOR.get(cell[0], (255, 255, 255))
                    except IndexError:
                        pass
    #instances = skimage.color.label2rgb(labels, initData, kind='avg')
    return segmentedImg, plot_markers, instances


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataName = sys.argv[1]

        if '.' in dataName:
            initData = cv2.imread(dataName)
            '''if initData.size > 512*512:
                initData = cv2.resize(initData, (512, 512))'''
        else:
            initData = skimage.data.astronaut()

    multiC = True
    colored = True
    if len(sys.argv) > 2 and sys.argv[2] == 'bw':
        initData = rgb2gray(initData)
        multiC = False
        colored = False
    #graph cuts
    #imageSegmentation(NAMESTODATA[dataName], algo='ap')

    #random Walker
    if not 'initData' in locals():
        print('Usage : give the name of the input data in arg')
        exit(0)

    #run random walker
    #initData = introduceNoise(initData, 0.4, colored)
    data, markers = generateInput(initData, autoseed=False)
    print(markers[0][0])
    print(markers.shape)


    #markers = np.loadtxt('./images/marked/markerselephant.txt')
    #cv2.imwrite('./images/marked/markers' + dataName.split('/')[-1].split('.')[0] + '.jpg', markers)
    '''toSave = np.zeros((markers.shape[0], markers.shape[1]), dtype=np.uint8)
    for i, raw in enumerate(markers):
        for j, cell in enumerate(raw):
            toSave[i][j] = cell[0]

    np.savetxt('./images/marked/markers' + dataName.split('/')[-1].split('.')[0] + '.txt', toSave, fmt='%i')
    #markers = np.loadtxt('./images/marked/markers' + dataName.split('/')[-1].split('.')[0] + '.txt')'''
    '''tempmark = np.loadtxt('markerstreet.txt')
    markers = np.zeros((tempmark.shape[0], tempmark.shape[1], 3), dtype=np.uint8)
    for i, raw in enumerate(tempmark):
        for j, cell in enumerate(raw):
            if cell> 0:
                markers[i][j][0] = cell'''

    '''markers = cv2.imread('./images/marked/hemen_markers2.jpg')
    print(markers.shape)
    data = cv2.resize(data, dsize=markers.shape[:2][::-1])
    print(data.shape)'''



    dispparam = 130 #default beta value
    beta = -1 #will be updated...
    while dispparam != beta:
        beta = dispparam # ...here
        rwmode = 'bf' if len(data.shape) < 3 else 'cg'
        print('Running RW with beta = {}'.format(beta))
        begin = time.time()
        labels = random_walker(data, markers, mode=rwmode, beta=beta)
        end = time.time()
        print('LABELS SHAPE',labels.shape)
        print("Running time : {:.4f}".format(end - begin))

        '''seuil = skimage.filters.threshold_otsu(data)
        labels[data < seuil] = 1
        labels[data >= seuil] = 2'''

        # display results
        segmented, plMarkers, instances = toPlot(data, markers, labels)

        #def plot_res(data, markers, labels, instances, algo, segparam):
        dispparam = plot_res(data,
                             dataName.split('/')[-1].split('.')[0],
                             plMarkers,
                             segmented,
                             labels,
                             instances,
                             'RW',
                             beta)

#    cv2.imwrite('./markers.jpg', plMarkers)
#    cv2.imwrite('/home/tdambrin/Documents/insa/pir/images/marked/markers.jpg', plMarkers)


    #run watershed
    '''compactness = 0
    print('Running RW with compactness = {}'.format(compactness))
    begin = time.time()
    labelsW = watershed(data, markers=markers, compactness=compactness)
    end = time.time()
    print("Running time : {:.4f}".format(end - begin))
    segmentedW, plMarkersW = toPlot(initData, markers, labelsW)
    dispparam = plot_res(data, plMarkers, segmentedW, 'WD', compactness)'''

    '''
    dispparam = 0 #default compactness value
    compactness = -1
    while dispparam != compactness:
        if dispparam < 0:
            raise ValueError('Cannot segment with compactness < 0')
        compactness = dispparam
        print('Running WS with compactness = {}'.format(compactness))
        begin = time.time()
        labelsW = watershed(data, markers=markers, compactness=compactness)
        end = time.time()
        print("Running time : {:.4f}".format(end - begin))
        segmentedW, plMarkersW, instancesW = toPlot(data, markers, labelsW)
        #def plot_res(data, markers, labels, instances, algo, segparam):
        dispparam = plot_res(data,
                             dataName.split('/')[-1].split('.')[0],
                             plMarkersW,
                             segmentedW,
                             labelsW,
                             instancesW,
                             'WD',
                             compactness)
'''