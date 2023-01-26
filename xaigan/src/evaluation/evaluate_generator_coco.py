import numpy as np
import os
import torch
import argparse
import random, string,json
import matplotlib.pyplot as plt

from PIL import Image
from copy import deepcopy
from utils.vector_utils import noise_coco, values_target
from utils.explanation_utils import extract_explanation
from torch.nn import functional as F

from models.generators import *
from models.text_embedding_models import *
from models.encoders import *
from models.configurations import configurations

from captum.attr import visualization as viz
from captum.attr import DeepLift, Saliency, IntegratedGradients, ShapleyValueSampling, Lime,LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
# from lime import lime_image
# from skimage.segmentation import mark_boundaries



def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--generator", default="./../results/MSCOCONormal/generator.pt", help="path of the file")
    parser.add_argument("-f2", "--discriminator", default="./../results/MSCOCONormal/discriminator.pt", help="path of the file")
    parser.add_argument("-n", "--number_of_samples",
                        type=int, default=2048,
                        help="number of samples to generate")

    args = parser.parse_args()
    sampling_args = {
            'generator' :Generator_Encoder_Net_CIFAR10,
            "text_emb_model"  :Glove_Embbeding,
            'use_captions'  :True,
            'dataset'       :'flowers-102',
            'noise_emb_sz'  :100,
            'text_emb_sz'   :50,
            'Encoder_emb_sz': (100+50)//2,
            'pretrained_generatorPath': args.generator,
            "pretrained_discriminatorPath": args.discriminator,
        }
    calculate_metrics_coco(sampling_args, numberOfSamples=args.number_of_samples)

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
    if(captions==None) : return None
    N=len(captions)
    if  dataset=='mscoco':  #using MSCOC-2014 val set captions
        return [captions[random.randint(0, N)]['caption'] for i in range(0,number)]
    elif dataset=='flowers-102': #using same dataset of flowers-102 captions
        return [captions[random.randint(0, N)] for i in range(0,number)]
        
def calculate_metrics_coco(args,numberOfSamples=2048):
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
    generate_samples_coco(numberOfSamples,args, path_output=folder)
    return

def generate_samples_coco(number,args,path_output):
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
    

    generatorArch = args['generator']
    gen_path = args['pretrained_generatorPath']
    text_emb_model = args["text_emb_model"]
    dataset = args['dataset']
    noise_emb_sz = args['noise_emb_sz']
    text_emb_sz = args['text_emb_sz']
    Encoder_emb_sz = args['Encoder_emb_sz']
    explainable = args["explainable"]
    explanation_type = args["explanationType"]
    explanation_types = args["explanationTypes"]
    discriminatorArch = args["discriminator"]
    disc_path = args['pretrained_discriminatorPath']
    use_captions = args['use_captions']
    use_captions_only = args['use_captions_only']
    use_one_caption = args['use_one_caption']

    # Declare & intialize Models
    if use_captions:
        random_texts = read_random_captionsFile(dataset)
        random_texts = get_random_text(number,random_texts,dataset) #using MSCOC-2014 val set captions
        text_emb_model = text_emb_model()
        text_emb_model.eval() 
        text_emb_model.device='cpu'

        if use_captions_only==False: #noise+captions
                generator = generatorArch(
                    noise_emb_sz=noise_emb_sz,
                    text_emb_sz=text_emb_sz,
                    n_features=Encoder_emb_sz)    
        else: #captions only
            generator = generatorArch(n_features=text_emb_sz)
    else:    #noise only
        generator = generatorArch(n_features=noise_emb_sz)
        # generator.to('cpu')

    discriminator = None
    explainer = None

    if explainable:
        discriminator = discriminatorArch()
        discriminator.load_state_dict(torch.load(disc_path, map_location=lambda storage, loc: storage)['model_state_dict'])
        discriminator.eval()


    # load saved weights for generator  & intialize Models
    generator.load_state_dict(torch.load(gen_path, map_location=lambda storage, loc: storage)['model_state_dict'])
    generator.eval()
    
    
    #todo: make sure we are in evualuation mode, batch Normaliation inference variables need to be used

    for i in range(number):
        noise = noise_coco(1, False)

        if use_captions:
            print(random_texts[i])
            random_text_emb = text_emb_model.forward([random_texts[i]]).detach()

            if use_captions_only==False: #noise + captions
                dense_emb = torch.cat((random_text_emb,noise.reshape(1,-1)), 1)
            else: #captions only
                dense_emb = random_text_emb[:,:,np.newaxis, np.newaxis]
        else: #use_captions==False
            dense_emb=noise
        
        sample = generator(dense_emb).detach()
        sample_image = sample.squeeze(0).numpy()
        sample_image = np.transpose(sample_image, (1, 2, 0))
        sample_image = ((sample_image/2) + 0.5) * 255
        sample_image = sample_image.astype(np.uint8)

        if explainable:
            disc_score= discriminator(sample)
            disc_result = "Fake" if disc_score < 0.5 else "Real"
            disc_score = round(disc_score.item(),4) # p = score of being Real. here we want all generated images to be fake, ie <0.5
            sample.requires_grad = True
            sample_bk = sample
            explanation = None
            for type in explanation_types:
                #reset the gradients of model's weights and the input.
                discriminator.zero_grad()
                sample=sample_bk
                explanation = extract_explanation(discriminator,sample,type)
                explanation = np.transpose(explanation.squeeze().cpu().detach().numpy(), (1, 2, 0))

                # Overlay explaination ontop of the original image
                figure, axis = viz.visualize_image_attr(explanation, sample_image, method= "blended_heat_map", sign="all",
                                                    show_colorbar=True,title=f"{type}, {disc_result}, D(G(z))={disc_score}")
                # Then Save explanation
                figure.savefig(f'{path_output}/{i}_{type}.jpg')
                
            sample.requires_grad = False


        # Save the original image
        figure, axis = viz.visualize_image_attr(None, sample_image, method="original_image", title="Original Image")
        figure.savefig(f'{path_output}/{i}.jpg')        
        
    return

if __name__ == "__main__":
    main()