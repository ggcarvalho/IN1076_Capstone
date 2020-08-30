from image import *
from argparse import ArgumentParser

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('path', type = str, help  = 'path to image')
    parser.add_argument('name', type = str, help  = 'name: grayscale, halftone, mean, gaussian, sharpen, laplacian, emboss,motion')
    parser.add_argument('save', type = str2bool, help = 'Boolean to save output. See README.md forl ist o allowed values.')
    return parser

def main():
    parser = arg_parser()
    args   = parser.parse_args()

    path = args.path
    name = args.name
    save = args.save

    proc_image(path, name, save)

if __name__ == '__main__':
    main()
