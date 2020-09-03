import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pylab import rcParams
from functools import reduce
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
        depth = len(image[0,0])
    except:
        depth = 1
    return height, width, depth

def is_grayscale(image):
    if get_shape(image)[2] == 3:
        return False
    elif get_shape(image)[2] == 1:
        return True
    else:
        raise Exception("Sorry, something is wrong!")

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
        height, width, _ = get_shape(image)
        gray_image       = zeros(height, width, 1)

        for i in tqdm(range(height), desc = "grayscale"):
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
            plt.imshow(image, cmap = "gray")
            plt.axis("off")
            plt.show()
        return image

def get_grayscale_image_range(image):
    if is_grayscale(image):
        image_min = reduce(min, map(min, image))
        image_max = reduce(max, map(max, image))
        return image_min, image_max
    else:
        raise Exception("This function works for grayscale images only!")

def adjust(image, new_min, new_max):
    image_min, image_max = get_grayscale_image_range(image)
    h, w, d = get_shape(image)
    adjusted = zeros(h, w, d)
    for i in tqdm(range(h), desc = "Adjusting the image"):
        for j in range(w):
            adjusted[i, j] = int((image[i, j] - image_min)*((new_max - new_min)/(image_max - image_min)) + new_min)
    return adjusted

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
    gray      = convert_grayscale(image, False, False)
    adjusted  = adjust(gray, 0, 9)
    m         = gen_halftone_masks()

    height, width, _ = get_shape(image)
    halftoned        = zeros(3*height, 3*width, 1)
    for j in tqdm(range(height), desc = "halftone"):
        for i in range(width):
            index = adjusted[j, i]
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

           "sharpen"   : np.array([[0 , -1,  0],
                                   [-1,  5, -1],
                                   [0 , -1,  0]]),

           "laplacian" : np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]]),

           "emboss"    : np.array([[-2, -1, 0],
                                   [-1,  1, 1],
                                   [ 0,  1, 2]]),

           "motion"    : np.array([[1/9, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1/9, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1/9, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1/9, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1/9, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1/9, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1/9, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1/9, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1/9]]),

           "y_edge"    : np.array([[1 ,  2, 1],
                                   [0 ,  0, 0],
                                   [-1, -2,-1]]),

           "x_edge"    : np.array([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]]),

            "brighten" : np.array([[0,  0 , 0],
                                   [0, 1.25, 0],
                                   [0,  0 , 0]]),

            "darken"   : np.array([[0 ,  0  , 0],
                                   [0 , 0.75, 0],
                                   [0 ,  0  , 0]]),

            "identity" : np.array([[0, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])}

def apply_kernel(image, kernel, save):
    kernel_matrix = kernels.get(kernel)
    dim           = len(kernel_matrix)
    center        = (dim - 1)//2

    height, width, _ = get_shape(image)

    if not is_grayscale(image):
        picture = zeros(height, width, 3)

        for y in tqdm(range(height), desc = kernel):
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

                r, g, b = map(int,  [redc, greenc, bluec])
                r, g, b = map(clip, [r, g, b])

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

        for y in tqdm(range(height), desc = kernel):
            for x in range(width):

                aux = zeros(dim, dim, 1)
                for i in range(dim):
                    for j in range(dim):
                        aux[i , j] = image[ (y - center + j)%height, (x - center + i)%width]

                gray = 0
                for i in range(dim):
                    for j in range(dim):
                        gray += aux[i, j]*kernel_matrix[i, j]

                pxl_intensity = round(gray)
                pxl_intensity = clip(pxl_intensity)
                picture[y, x] = int(pxl_intensity)
        if save:
            cv2.imwrite(kernel + ".png", picture)

        plt.imshow(picture, cmap = "gray")
        plt.axis("off")
        plt.show()
        return picture

def transpose(m):
    height, width, depth = get_shape(m)

    transposed = zeros(width, height, depth)
    for i in range(width):
        for j in range(height):
            transposed[i, j] = m[j, i]
    return transposed

def aux90(image):
    return transpose(image)[:,::-1]

def rot90(image, save):
    print("Rotating the image 90 degrees clockwise...")
    rot = aux90(image)
    if save:
        cv2.imwrite("rot90.png", rot)
    if is_grayscale(image):
        plt.imshow(rot, cmap = "gray")
        plt.axis("off")
        plt.show()
    else:
        plt.imshow(rot[:, :, ::-1])
        plt.axis("off")
        plt.show()
    return transpose(image)[:,::-1]

def rot180(image, save):
    print("Rotating the image 180 degrees clockwise...")
    rot = image[::-1, ::-1]
    if save:
        cv2.imwrite("rot180.png", rot)
    if is_grayscale(image):
        plt.imshow(rot, cmap = "gray")
        plt.axis("off")
        plt.show()
    else:
        plt.imshow(rot[:, :, ::-1])
        plt.axis("off")
        plt.show()
    return rot

def rotm90(image, save):
    print("Rotating the image 90 degrees counterclockwise...")
    rot = aux90(image[::-1, ::-1])
    if save:
        cv2.imwrite("rot270.png", rot)
    if is_grayscale(image):
        plt.imshow(rot, cmap = "gray")
        plt.axis("off")
        plt.show()
    else:
        plt.imshow(rot[:, :, ::-1])
        plt.axis("off")
        plt.show()
    return rot

def vert_flip(image, save):
    print("Flipping vertically...")
    flip = image[:, ::-1]
    if save:
        cv2.imwrite("vflip.png", flip)
    if is_grayscale(image):
        plt.imshow(flip, cmap = "gray")
        plt.axis("off")
        plt.show()
    else:
        plt.imshow(flip[:, :, ::-1])
        plt.axis("off")
        plt.show()
    return flip

def hor_flip(image, save):
    print("Flipping horizontally...")
    flip = image[::-1]
    if save:
        cv2.imwrite("hflip.png", flip)
    if is_grayscale(image):
        plt.imshow(flip, cmap = "gray")
        plt.axis("off")
        plt.show()
    else:
        plt.imshow(flip[:, :, ::-1])
        plt.axis("off")
        plt.show()
    return flip

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
             "identity"  : apply_kernel,
             "rot90"     : rot90,
             "rot180"    : rot180,
             "rotm90"    : rotm90,
             "hor_flip"  : hor_flip,
             "vert_flip" : vert_flip}

not_kernel = ["grayscale", "halftone", "rot90",
             "rot180", "rotm90", "vert_flip", "hor_flip"]

def proc_image(path, name, save):
    try:
        image    = cv2.imread(path, cv2.IMREAD_UNCHANGED|cv2.IMREAD_ANYDEPTH)
        function = functions.get(name)
        if name in not_kernel:
            function(image, save)
        else:
            function(image, name, save)
    except:
        raise Exception("\nSorry, something is wrong!")

def main():
    SAVE      = str2bool(sys.argv[1])
    path      = "test1.png"
    image     = cv2.imread(path, cv2.IMREAD_UNCHANGED|cv2.IMREAD_ANYDEPTH)

    gray      = convert_grayscale(image, SAVE)
    half      = halftone(image, SAVE)
    mean      = apply_kernel(image, "mean", SAVE)
    gaussian  = apply_kernel(image, "gaussian", SAVE)
    sharpen   = apply_kernel(image, "sharpen", SAVE)
    laplacian = apply_kernel(image, "laplacian", SAVE)
    emboss    = apply_kernel(image, "emboss", SAVE)
    motion    = apply_kernel(image, "motion", SAVE)
    x_edge    = apply_kernel(image, "x_edge", SAVE)
    y_edge    = apply_kernel(image, "y_edge", SAVE)
    brighten  = apply_kernel(image, "brighten", SAVE)
    darken    = apply_kernel(image, "darken", SAVE)
    identity  = apply_kernel(image, "identity", SAVE)
    r90       = rot90(image, SAVE)
    r180      = rot180(image, SAVE)
    rm90      = rotm90(image, SAVE)
    hflip     = hor_flip(image, SAVE)
    vflip     = vert_flip(image, SAVE)

    print("Done!")
if __name__ == "__main__":
    main()