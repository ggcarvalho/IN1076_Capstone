import cv2, sys
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from argparse import ArgumentTypeError

rcParams['figure.figsize'] = 8, 8

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_shape(image):
    height  = len(image)
    width   = len(image[0])

    try:
        depth   = len(image[0,0])
    except:
        depth = 1

    return height, width, depth

def is_grayscale(image):
    if len(get_shape(image)) == 3:
        return False
    return True

def clip(a):
    if a < 0:
        return 0

    elif a > 255:
        return 255

    else:
        return a

def get_luminance(r, g, b):
    return 0.299*r + 0.587*g + 0.114*b

def zeros(height, width, depth):
    return np.array([[[0 for k in range(depth)] for j in range(width)] for i in range(height)]) if depth != 1\
          else np.array([[0 for j in range(width)] for i in range(height)])

def convert_grayscale(image, save, show = True):
    if not is_grayscale(image):

        height     = get_shape(image)[0]
        width      = get_shape(image)[1]
        gray_image = zeros(height, width, 1)

        for i in range(height):
            for j in range(width):
                r, g, b   = image[i, j, 2], image[i, j, 1], image[i, j, 0]
                luminance = get_luminance(r, g, b)

                gray_image[i, j] = luminance
        if save:
            cv2.imwrite("gray.png", gray_image)
        if show:
            plt.imshow(gray_image, cmap = "gray")
            plt.axis("off")
            plt.show()
        return np.array(gray_image)
    else:
        if show:
            plt.imshow(gray_image, cmap = "gray")
            plt.axis("off")
            plt.show()
        return image

def min_max(image, new_min, new_max):
    return ((image - image.min())*( (new_max - new_min)/(image.max() - image.min()) ) + new_min).astype(np.uint8)

def gen_halftone_masks():
    m = zeros(3, 3, 10)

    m[:, :, 1] = m[:, :, 0]
    m[0, 1, 1] = 1

    m[:, :, 2] = m[:, :, 1]
    m[2, 2, 2] = 1

    m[:, :, 3] = m[:, :, 2]
    m[0, 0, 3] = 1

    m[:, :, 4] = m[:, :, 3]
    m[2, 0, 4] = 1

    m[:, :, 5] = m[:, :, 4]
    m[0, 2, 5] = 1

    m[:, :, 6] = m[:, :, 5]
    m[1, 2, 6] = 1

    m[:, :, 7] = m[:, :, 6]
    m[2, 1, 7] = 1

    m[:, :, 8] = m[:, :, 7]
    m[1, 0, 8] = 1

    m[:, :, 9] = m[:, :, 8]
    m[1, 1, 9] = 1

    return m

def halftone(image, save):
    gray = convert_grayscale(image, False, False)
    adjust = min_max(gray, 0, 9)
    m = gen_halftone_masks()
    halftoned = zeros(3*get_shape(adjust)[0], 3*get_shape(adjust)[1], 1)
    for j in range(get_shape(adjust)[0]):
        for i in range(get_shape(adjust)[1]):
            index = adjust[j, i]
            halftoned[3*j:3+3*j, 3*i:3+3*i] = m[:, :, index]

    halftoned = 255*halftoned
    if save:
        cv2.imwrite("halftone.png", halftoned)

    plt.imshow(halftoned, cmap = "gray")
    plt.axis("off")
    plt.show()
    return halftoned

kernels = {"mean"      : np.array([[1/9, 1/9, 1/9],
                                   [1/9, 1/9, 1/9],
                                   [1/9, 1/9, 1/9]]),

           "gaussian"  : np.array([[1/16, 2/16, 1/16],
                                   [2/16, 4/16, 2/16],
                                   [1/16, 2/16, 1/16]]),

           "sharpen"   : np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]]),

           "laplacian" : np.array([[-1, -1, -1],
                                   [-1, 8, -1],
                                   [-1, -1, -1]]),

           "emboss"    : np.array([[-2, -1, 0],
                                   [-1, 1 , 1],
                                   [0, 1, 2]]),

           "motion"    : np.array([[1/9, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1/9, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1/9, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1/9, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1/9, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1/9, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1/9, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1/9, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1/9]]),

           "y_edge"    : np.array([[1, 2, 1],
                                   [0, 0 ,0],
                                   [-1, -2, -1]]),

           "x_edge"    : np.array([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]]),

            "brighten" : np.array([[0, 0, 0],
                                    [0, 1.2, 0],
                                    [0, 0, 0]]),

            "darken"   : np.array([[0, 0, 0],
                                    [0, 0.75, 0],
                                    [0, 0, 0]]),

            "identity" : np.array([[0, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])}

def apply_kernel(image, kernel, save):
    kernel_matrix = kernels.get(kernel)
    dim           = len(kernel_matrix)
    center        = (dim - 1)//2

    width   = get_shape(image)[1]
    height  = get_shape(image)[0]

    if not is_grayscale(image):
        picture = zeros(height, width, 3)

        for y in range(height):
            for x in range(width):

                red = zeros(dim, dim, 1)
                for i in range(dim):
                    for j in range(dim):
                        red[i , j] = image[ (y - center + j)%height, (x - center + i)%width, 2]

                green = zeros(dim, dim, 1)
                for i in range(dim):
                    for j in range(dim):
                        green[i , j] = image[ (y - center + j)%height, (x - center + i)%width, 1]

                blue = zeros(dim, dim, 1)
                for i in range(dim):
                    for j in range(dim):
                        blue[i , j] = image[ (y - center + j)%height, (x - center + i)%width, 0]

                redc, greenc, bluec = 0, 0, 0

                for i in range(dim):
                    for j in range(dim):
                        redc   += red[i, j]*kernel_matrix[i, j]
                        greenc += green[i, j]*kernel_matrix[i, j]
                        bluec  += blue[i, j]*kernel_matrix[i, j]

                r = round(redc)
                g = round(greenc)
                b = round(bluec)

                r, g, b = clip(r), clip(g), clip(b)

                picture[y, x, 2] = r
                picture[y, x, 1] = g
                picture[y, x, 0] = b
        if save:
            cv2.imwrite(kernel + ".png", picture)
        plt.imshow(picture[:, :, ::-1])
        plt.axis("off")
        plt.show()
        return picture
    else:
        picture = zeros(height, width, 1)

        for y in range(height):
            for x in range(width):

                aux = zeros(dim, dim, 1)
                for i in range(dim):
                    for j in range(dim):
                        aux[i , j] = image[ (y - center + j)%height, (x - center + i)%width]

                intensity = 0
                for i in range(dim):
                    for j in range(dim):
                        intensity += aux[i, j]*kernel_matrix[i, j]

                pxl_intensity = round(intensity)
                picture[y, x] = pxl_intensity
        if save:
            cv2.imwrite(kernel + ".png", picture)

        plt.imshow(picture[:, :, ::-1])
        plt.axis("off")
        plt.show()
        return picture

functions = {"grayscale" : convert_grayscale,
             "halftone"  : halftone,
             "mean"      : apply_kernel,
             "gaussian"  : apply_kernel,
             "sharpen"   : apply_kernel,
             "laplacian" : apply_kernel,
             "emboss"    : apply_kernel,
             "motion"    : apply_kernel,
             "x_edge"    : apply_kernel,
             "y_edge"    : apply_kernel,
             "brighten"  : apply_kernel,
             "darken"    : apply_kernel,
             "identity"  : apply_kernel}

def proc_image(path, name, save):
    try:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED|cv2.IMREAD_ANYDEPTH)
        function = functions.get(name)
        if name == "grayscale" or name == "halftone":
            function(image, save)
        else:
            function(image, name, save)
    except:
        print("\nSomething went wrong! Please check the image path and filter name!\n\nRun:\npython proc_image.py -h\nfor help!")

def main():
    path      = "test1.jpeg"
    image     = cv2.imread(path, cv2.IMREAD_UNCHANGED|cv2.IMREAD_ANYDEPTH)

    gray      = convert_grayscale(image, True)
    half      = halftone(image, True)
    mean      = apply_kernel(image, "mean", True)
    gaussian  = apply_kernel(image, "gaussian", True)
    sharpen   = apply_kernel(image, "sharpen", True)
    laplacian = apply_kernel(image, "laplacian", True)
    emboss    = apply_kernel(image, "emboss", True)
    motion    = apply_kernel(image, "motion", True)

if __name__ == "__main__":
    main()