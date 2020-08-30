# IN1076 Capstone

Final project for IN1076 @ CIN - UFPE, 2020.1.

This project aims at implementing some image processing tools without relying on built-in functions found in several libraries. We use:

- Open CV: read and write images.

- Argparse and Sys: parse command line arguments.

- Matplotlib: display output.

- Pylab: format the displayed output.

- Numpy: used to store arrays, uint8 type casting, min-max calculation.

There are two Python files, `image.py` and `proc_image.py`, the former containing all the implementation needed and a test client, while the latter is the main application.

### Disclaimer

The program is not optimized, performance-wise. Therefore, for sufficiently large pictures this process takes a while.

## Using the program

To use this program run:

    python proc_image.py <path to image> <process name> <save>

### Example

    python proc_image.py ~/Desktop/cat.jpg gaussian False

The processed image will always be displayed when the processing is done.

## Using the test client

Run

    python image.py <save>

### Example

Run

    python image.py 0

execute the test client without saving the outputs.

## Remark

List of allowed boolean values for `<save>`:

`True`: true, yes, t, y, 1, True, TRUE. In general, if v.lower() == true, you're fine.

`False`: false, no, f, n, 0, False, FALSE. In general, if v.lower() == false, you're fine.

## Tools

The image processing tools avaiable and their respective names (you should use theses names when running the program) are given below:

- Grayscale filter (`grayscale`): converts an RGB image into grayscale using the luminance of a pixel. The luminance is calculated using l = 0.299r + 0.587g + 0.114b where r, g, and b are the pixel values for the red, green, and blue channel, respectively.

- Halftone (`halftone`): converts the range of a grayscale image to [0, 9] and for each pixel value performs a mapping acording to the following image from this [reference](http://www.imageprocessingplace.com/DIP-3E/dip3e_student_projects.htm#02-01). Due to this mapping, halftoned images have three times the width and three times the heght of the original image.

![Halftone map](halftone_map.png)

- Mean blur (`mean`): takes the average of 3 x 3 regions.

- Gaussian blur (`gaussian`): takes an weighted average of a 3 x 3 region using a gaussian function.

- Sharpen (`sharpen`): sharpens the image. Formally, substracts the 4-neighboor laplacian from the original image.

- Laplacian (`laplacian`): returns the 8-neighboor laplacian applied to the image.

- Emboss (`emboss`): Enhance image emboss.

- Motion blur (`motion`): Blurs the image as if it is moving.

- Edge detectors (`y_edge`, `x_edge`): Sobel filters to detect vertical and horizontal edges, respectively.

- Brighten (`brighten`): Brightens the image in 20%.

- Darken (`darken`): Darkens the image in 25%.

- Identity (`identity`): Returns the original image.

## Convolution / Cross-correlation

The function apply_kernel in `image.py` implements the cross-correlation. It is similar to a convolution, without needing to "rotate" the kernel matrices. All of the kernel matrices are already "rotated". In that case, the cross-correlation with the given kernel is, by definition, the convolution needed to process the image.
![convolution](conv.gif)