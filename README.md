# IN1076 Capstone

To use this program run:

    python proc_image.py <path to image> <filter name> <save>

### Example

    python proc_image.py ~/Desktop/cat.jpg gaussian False

The processed image will always be displayed when the processing is done.

## Test client

Run

    python image.py

## Filters

The image processing filters avaiable and their respective names (you should use theses names when running the program) are gven below:

- Grayscale filter (`grayscale`): converts an RGB image into grayscale using the luminance of a pixel. The luminance is calculated using l = 0.299*r + 0.587*g + 0.114*b where r, g, and b are the pixel values for the red, green, and blue channel, respectively.

- Halftone (`halftone`): converts the range of a grayscale image to [0, 9] and for each pixel value performs a mapping acording to the following image from this [reference](http://www.imageprocessingplace.com/DIP-3E/dip3e_student_projects.htm#02-01). Due to this mapping, halftoned images have three times the width and three times the heght of the original image.

![Halftone map](halftone_map.png)

- Mean blur (`mean`): takes the average of a $`3\times 3`$ region.

- Gaussian blur (`gaussian`): takes an weighted average of a $3\times 3$ region using a gaussian function.

- Sharpen (`sharpen`): sharpens the image. Formally, substracts the $4-$neighboor laplacian from the original image.

- Laplacian (`laplacian`): returns the $8-$neighboor laplacian applied to the image.

- Emboss (`emboss`): Enhance image emboss.

- Motion blur (`motion`): Blurs the image as if it is moving.

- Edge detectors (`y_edge`, `x_edge`): Sobel filters to detect vertical and horizontal edges, respectively.

## Convolution / Cross-correlation

The function apply_kernel in `image.py` implements the cross-correlation. It is similar to a convolution, without needing to "rotate" the kernel matrices. All of the kernel matrices are already "rotated". In that case, the cross-correlation with the given kernel is, by definition, the convolution needed to process the image.