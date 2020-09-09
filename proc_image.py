from image import *
from argparse import ArgumentParser

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("path", type = str, help  = "Path to image. Example: ~/Desktop/cats.png")

    parser.add_argument("name", type = str, help  = "processing name: grayscale, halftone, mean, gaussian,\
                                                                      sharpen, laplacian, emboss, motion,\
                                                                      x_edge, y_edge, brighten, darken,\
                                                                      identity, rot90, rot180, rotm90,\
                                                                      vert_flip, hor_flip, downscale, negative")

    parser.add_argument("save", type = str2bool, help = "Boolean to save output. See README.md\
                                                         for list of allowed values.")
    return parser

def main():
    parser = arg_parser()
    args   = parser.parse_args()

    path = args.path
    name = args.name
    save = args.save

    proc_image(path, name, save)

if __name__ == "__main__":
    main()
