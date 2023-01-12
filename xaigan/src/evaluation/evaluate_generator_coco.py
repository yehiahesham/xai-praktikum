import numpy as np
import os
import torch
import argparse
import random, string,json
from PIL import Image
from utils.vector_utils import noise_coco
from models.generators import * 
from models.text_embedding_models import *
from models.encoders import *
from models.configurations import configurations




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

def read_random_captionsFile(dataset):    
    captions=None     
    if  dataset=='mscoco': 
        captions_json_path='./xaigan/src/evaluation/captions_val2014.json'
        with open(captions_json_path) as f:
            captions = json.load(f)
            captions=captions['annotations']   
    elif dataset=='flowers-102': 
        CFG  = configurations["datasets"]["flowers-102"]        
        captions_json_path = CFG["path"]+"/"+"flowers102_captions.json"
        with open(captions_json_path) as f:
            captions_array = json.load(f).values()
            captions = [caption for Captions in captions_array for caption in Captions]
    return captions

def get_random_text(number,captions,dataset):
    N=len(captions)
    if  dataset=='mscoco':  #using MSCOC-2014 val set captions
        return [captions[random.randint(0, N)]['caption'] for i in range(0,number)]
    elif dataset=='flowers-102': #using same dataset of flowers-102 captions
        return [captions[random.randint(0, N)] for i in range(0,number)]
        
def calculate_metrics_coco(path, generatorArch, use_captions,dataset,TextEmb_Arch=None,numberOfSamples=2048):
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
    generate_samples_coco(numberOfSamples,generatorArch,TextEmb_Arch,use_captions,dataset, path_model=path, path_output=folder)
    return


def generate_samples_coco(number,generatorArch,TextEmb_Arch,use_captions,dataset,path_model, path_output):
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

    # Declare & intializ Variables
    random_texts = read_random_captionsFile(dataset)
    random_texts = get_random_text(number,random_texts,dataset) #using MSCOC-2014 val set captions
    noise_emb_sz,text_emb_sz=100,768
    Encoder_emb_sz =(noise_emb_sz+text_emb_sz)//2 #Hyper-paramter (encoder output/ generator input)
    
    # Declare & intialize Models
    if use_captions:
        text_emb_model = TextEmb_Arch()
        generator = generatorArch(
            noise_emb_sz=noise_emb_sz,
            text_emb_sz=text_emb_sz,
            n_features=Encoder_emb_sz)
        text_emb_model.eval() 
        text_emb_model.device='cpu'
    else:
        generator = generatorArch(n_features=noise_emb_sz)

    # load saved weights for generator  & intialize Models
    generator.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage)['model_state_dict'])
    generator.eval()
    
    
    #todo: make sure we are in evualuation mode, batch Normaliation inference variables need to be used

    for i in range(number):
        noise = noise_coco(1, False)
        if use_captions:
            print(random_texts[i])
            random_text_emb = text_emb_model.forward([random_texts[i]]).detach()
            dense_emb = torch.cat((random_text_emb,noise.reshape(1,-1)), 1)            
        else:
            dense_emb=noise
        
        sample = generator(dense_emb).detach().squeeze(0).numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample = ((sample/2) + 0.5) * 255
        sample = sample.astype(np.uint8)
        image = Image.fromarray(sample)
        image.save(f'{path_output}/{i}.jpg')
    
    return


if __name__ == "__main__":
    main()