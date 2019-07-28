# Exercise 1: Image Representations & Intensity Transformations &Quantization

**Overview**
The Assignment's goals are :

* Loading grayscale and RGB image representations.
* Displaying figures and images.
* Transforming RGB color images back and forth from the YIQ color space.
* Performing intensity transformations: histogram equalization.
* Performing optimal quantization


we made a function converting rgb images to greyscale images and vice versa


```Python
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

```



**Example** : image convertion from rgb to grey
>Lui the rabbit
![](https://github.com/Sniryefet/Image-Processing-Assignment_1/blob/master/pictures/rgb.PNG)
![](https://github.com/Sniryefet/Image-Processing-Assignment_1/blob/master/pictures/to_grey.PNG)

## Histograms
you can also see the picture's histogram (pixels intensity)


![](https://github.com/Sniryefet/Image-Processing-Assignment_1/blob/master/pictures/CDF_HISTOGRAM.PNG)


## Histogram Equalization
The program is also capable of improving picture's contrast using uniform dispersal of all grey levels


![](https://github.com/Sniryefet/Image-Processing-Assignment_1/blob/master/pictures/lol.jpg)
