import numpy as np
import os
import torch
import argparse
import random, string
from PIL import Image
from utils.vector_utils import noise_coco
from models.generators import * 
from models.text_embedding_models import RobertaClass
from models.encoders import *


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

def read_random_captionsFile(captions_json_path='./xaigan/src/evaluation/captions_val2014.json'):
    import random,json
    captions=None
    with open(captions_json_path) as f:
        captions = json.load(f)
        captions=captions['annotations']   
    return captions

def get_random_text(number,captions):
    N=len(captions)
    return [captions[random.randint(0, N)]['caption'] for i in range(0,number)]
            
    

    
        

def calculate_metrics_coco(path, generatorArch, numberOfSamples=2048):
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
    generate_samples_coco(numberOfSamples,generatorArch, path_model=path, path_output=folder)
    return


def generate_samples_coco(number,generatorArch, path_model, path_output):
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
    # random_texts = read_random_captionsFile()
    # random_texts = get_random_text(number,random_texts) #using MSCOC-2014 val set captions
    noise_emb_sz,text_emb_sz=100,768
    # Encoder_emb_sz =(noise_emb_sz+text_emb_sz)//2 #Hyper-paramter

    #Declare & load Models' weights
    # text_emb_model = RobertaClass()
    # generator = Encoder_GeneratorNet_TEXT2IMG_MSCOCO(
    #         noise_emb_sz = noise_emb_sz,
    #         text_emb_sz  = text_emb_sz,
    #         n_features   = Encoder_emb_sz)
    # generator = GeneratorNetMSCOCO(n_features=noise_emb_sz
    generator = generatorArch(n_features=noise_emb_sz)
    generator.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage)['model_state_dict'])
    
    # EmbeddingEncoder_old_mode=EmbeddingEncoder.training
    # generator_old_mode=generator.training
    generator.eval()
    # text_emb_model.eval() 
    # text_emb_model.device='cpu'
    #todo: make sure we are in evualuation mode, batch Normaliation inference variables need to be used
    #todo: make sure of the shapes of random_text_emb,noise,dense_emb if they need reshaping
    #todo: randomly pick or generate a better rand sentence 
    for i in range(number):
        noise = noise_coco(1, False)
        # print(random_texts[i])
        # random_text_emb = text_emb_model.forward([random_texts[i]]).detach() 
        # dense_emb = torch.cat((random_text_emb,noise.reshape(1,-1)), 1)
        # dense_emb = dense_emb.reshape(1,-1)
        dense_emb=noise
        sample = generator(dense_emb).detach().squeeze(0).numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample = ((sample/2) + 0.5) * 255
        sample = sample.astype(np.uint8)
        image = Image.fromarray(sample)
        image.save(f'{path_output}/{i}.jpg')
    
    # if EmbeddingEncoder_old_mode:  EmbeddingEncoder.train()
    # if generator_old_mode:         generator_old_mode.train()
    return


if __name__ == "__main__":
    main()