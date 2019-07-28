import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from scipy import misc
from PIL import Image
from typing import List

# 3.1 Reading an image into a given representation
def imReadAndConvert(filename: str, representation: int)-> np.ndarray:
    im = cv2.imread(filename)
    np_im = cv2.normalize(im.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    if len(np_im.shape) < 3 and representation == 2:
        print(1)
        np_im = gray2rgb(im)
    elif len(np_im.shape) == 3 and representation == 1:
        print(3)
        np_im = rgb2gray(np_im)

    return np_im

def rgb2gray(rgb):
     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
     return gray


def gray2rgb(gray):
    rgb = list()
    # how to do gray.color
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = gray.color, gray.color, gray.color
    return rgb

#3.2 Displaying an image
def imDisplay(filename:str, representation:int):
    np_im = imReadAndConvert(filename, representation)
    img = Image.fromarray(np_im)
    plt.imshow(img)
    plt.show()

# 3.3 Transforming an RGB image to YIQ color space
def transformRGB2YIQ(imRGB:np.ndarray)->np.ndarray:
    if len(imRGB.shape) == 3:

        yiq_ = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]])
        imYI = np.dot(imRGB, yiq_.T.copy())
        # plt.imshow(imYI)
        # plt.show()
        return imYI

def transformYIQ2RGB(imYIQ:np.ndarray)->np.ndarray:
    if len(imYIQ.shape) == 3:
        rgb_ = np.array([[1.00, 0.956, 0.623],
                        [1.0, -0.272, -0.648],
                        [1.0, -1.105, 0.705]])
        imRGB = np.dot(imYIQ, rgb_.T.copy())
        plt.imshow(imRGB)
        #plt.show()
        return imRGB

# 3.4 Histogram equalization
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
def histogramEqualize(imOrig:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray):
    temp = imOrig.copy()
    if len(imOrig.shape) == 3:
        imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
        imOrig = np.ceil(imOrig)
        imOrig = imOrig.astype('uint8')
        imOrig = cv2.cvtColor(imOrig, cv2.COLOR_BGR2RGB)
        imOrig = cv2.normalize(imOrig.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        imyiq = transformRGB2YIQ(imOrig)
        imOrig = imyiq[:, :, 0]
    # nirmul
    imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
    imOrig=np.ceil(imOrig)
    imOrig=imOrig.astype('uint8')
    hist, bins = np.histogram(imOrig.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(imOrig.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histogram'), loc='upper left')
    #plt.show()
    cdf_m = np.ma.masked_equal(cdf, 0)

    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    imOrig = imOrig.astype('uint8')
    img2 = cdf[imOrig]
    hist, bins = np.histogram(img2.flatten(), 256, [0, 256])
    #cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    hist, bins = np.histogram(img2.flatten(), 256, [0, 256])
    plt.plot(cdf_normalized, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='g')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histogram'), loc='upper left')
    plt.show()
    if len(temp.shape) == 3:
        imyiq[:, :, 0] = img2
        imOrig = transformYIQ2RGB(imyiq)

        imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
        imOrig = np.ceil(imOrig)
        imOrig = imOrig.astype('uint8')
        imOrig = cv2.cvtColor(imOrig, cv2.COLOR_RGB2BGR)


    return imOrig

#3.5 Optimal image quantization
def quantizeImage(imOrig:np.ndarray, nQuant:int, nIter:int)->(List[np.ndarray], List[float]):
    #imOrig
    temp = histogramEqualize(imOrig)
    img = Image.fromarray(temp)
    plt.imshow(img)
    plt.show()
    #nQuant
    temp = histogramEqualize(nQuant)
    img = Image.fromarray(temp)
    plt.imshow(img)
    plt.show()
    #nIter

if __name__ == "__main__":
    inr = imReadAndConvert("lui.jpg", 2)
    #transformYIQ2RGB(transformRGB2YIQ(inr))
    #np_im = cv2.normalize(inr.astype('double', None, 0.0, 1.0, cv2.NORM_MINMAX))

    temp = histogramEqualize(inr)
    img = Image.fromarray(temp)

    plt.imshow(img)
    plt.show()

    #img.save("test.jpg")
    #img = Image.open("test.jpg")
    #histogramEqualize(inr)

