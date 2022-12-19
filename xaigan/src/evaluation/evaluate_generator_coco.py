from utils.vector_utils import noise_coco
import torch
import argparse
import numpy as np
from models.generators import GeneratorNetMSCOCO
import os
from PIL import Image


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--file", default="./../results/MSCOCONormal/generator.pt", help="path of the file")
    parser.add_argument("-n", "--number_of_samples",
                        type=int, default=2048,
                        help="number of samples to generate")

    args = parser.parse_args()
    calculate_metrics_coco(path=args.file, numberOfSamples=args.number_of_samples)


def calculate_metrics_coco(path, numberOfSamples=2048):
    """
    This function is supposed to calculate metrics for coco.
    :param path: path of the generator model
    :type path: str
    :param numberOfSamples: number of samples to generate
    :type numberOfSamples: int
    :return: None
    :rtype: None
    """
    folder = f'{os.getcwd()}/tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    generate_samples_coco(number=numberOfSamples, path_model=path, path_output=folder)
    return


def generate_samples_coco(number, path_model, path_output):
    """
    This function generates samples for the coco GAN and saves them as jpg
    :param number: number of samples to generate
    :type number: int
    :param path_model: path where the coco generator is saved
    :type path_model: str
    :param path_output: path of the folder to save the images
    :type path_output: str
    :return: None
    :rtype: None
    """
    generator = GeneratorNetMSCOCO()
    generator.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage))
    for i in range(number):
        sample = generator(noise_coco(1, False)).detach().squeeze(0).numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample = ((sample/2) + 0.5) * 255
        sample = sample.astype(np.uint8)
        image = Image.fromarray(sample)
        image.save(f'{path_output}/{i}.jpg')
    return


if __name__ == "__main__":
    main()