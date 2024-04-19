import argparse

from tensorflow.keras.models import load_model
from image_utils import *


def main():
    parser = argparse.ArgumentParser(description="Process some flags.")

    # Add a required `input_file` argument
    parser.add_argument('input_file', type=str, help='The path to the input file.')
    parser.add_argument('-v', '--visualize', action='store_true', help='Enable visualization mode.')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug visualization mode.')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Use grayscale model.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Accessing the arguments
    vis = args.visualize
    debug = args.debug
    grayscale = args.grayscale

    # imgDir = '2024Simgs/'
    imgDir = args.input_file

    if grayscale:
        model = load_model('CNN_model_GS.h5')
    else:
        model = load_model('CNN_model.h5')

    if debug:
        vis = True

    test_data, lines, ext = get_test_data(imgDir, grayscale=grayscale)

    # test on test_data
    test_model(lines, imgDir, ext, test_data, model, visualize=vis, only_false=debug)


if __name__ == '__main__':
    main()

    