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
from captum.attr import visualization as viz
from captum.attr import DeepLiftShap, Saliency, IntegratedGradients, ShapleyValueSampling, Lime




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
    use_captions = args['use_captions']
    dataset = args['dataset']
    noise_emb_sz = args['noise_emb_sz']
    text_emb_sz = args['text_emb_sz']
    Encoder_emb_sz = args['Encoder_emb_sz']
    explainable = args["explainable"]
    explanation_type = args["explanationType"]
    explanation_types = args["explanationTypes"]
    discriminatorArch = args["discriminator"]
    disc_path = args['pretrained_discriminatorPath']

    # Declare & intialize Variables

    random_texts = None
    
    # Declare & intialize Models
    if use_captions:
        random_texts = read_random_captionsFile(dataset)
        random_texts = get_random_text(number, random_texts, dataset)  # using MSCOC-2014 val set captions
        text_emb_model = text_emb_model()
        generator = generatorArch(
            noise_emb_sz=noise_emb_sz,
            text_emb_sz=text_emb_sz,
            n_features=Encoder_emb_sz)
        text_emb_model.eval() 
        text_emb_model.device='cpu'
    else:
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
            dense_emb = torch.cat((random_text_emb,noise.reshape(1,-1)), 1)
        else:
            dense_emb=noise
        
        sample = generator(dense_emb).detach()
        sample_image = sample.squeeze(0).numpy()
        sample_image = np.transpose(sample_image, (1, 2, 0))
        sample_image = ((sample_image/2) + 0.5) * 255
        sample_image = sample_image.astype(np.uint8)

        if explainable:
            disc_score = discriminator(sample)
            disc_result = "Fake" if disc_score < 0.5 else "Real"
            sample.requires_grad = True
            for type in explanation_types:
                discriminator.zero_grad()
                if type == "saliency":
                    explainer = Saliency(discriminator)
                    explanation = explainer.attribute(sample)

                # elif type == "shap":
                #     explainer = DeepLiftShap(discriminator)
                #     explanation = explainer.attribute(sample, baselines=)

                elif type == "shapley_value_sampling":
                    explainer = ShapleyValueSampling(discriminator)
                    explanation = explainer.attribute(sample, n_samples=2)

                elif type == "Integrated_Gradients":
                    explainer = IntegratedGradients(discriminator)
                    explanation = explainer.attribute(sample, return_convergence_delta=True)

                explanation = np.transpose(explanation.squeeze().cpu().detach().numpy(), (1, 2, 0))

                # explanation = ((explanation / 2) + 0.5) * 255
                # explanation = explanation.astype(np.uint8)
                # explanation_image = Image.fromarray(explanation)
                # explanation_image.save(f'{path_output}/{i}_explanation.jpg')
                sample.requires_grad = False
                figure, axis = viz.visualize_image_attr(explanation, sample_image, method="blended_heat_map", sign="absolute_value",
                                             show_colorbar=True, title=f"{type}, {disc_result}")
                figure.savefig(f'{path_output}/{i}_{type}.jpg')


        # image = Image.fromarray(sample_image)

        # Save the original image
        figure, axis = viz.visualize_image_attr(None, sample_image, method="original_image", title="Original Image")
        figure.savefig(f'{path_output}/{i}.jpg')
        # image.save(f'{path_output}/{i}.jpg')
    
    return


if __name__ == "__main__":
    main()