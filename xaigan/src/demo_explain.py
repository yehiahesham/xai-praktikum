import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


import torch
import torchvision
from torchvision import transforms
from copy import deepcopy
from models.generators import *
from models.discriminators import *
from models.text_embedding_models import *
from models.encoders import *
from utils.explanation_utils import extract_explanation

from captum.attr import visualization as viz




def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--generator", default="./results/cifar-10/CIFAR10_only_SaliencyTrain/generator.pt", help="path of the file")
    parser.add_argument("-f2", "--discriminator", default="./results/cifar-10/CIFAR10_only_SaliencyTrain/discriminator.pt", help="path of the file")
    parser.add_argument("-n", "--number_of_samples",
                        type=int, default=2048,
                        help="number of samples to generate")

    args = parser.parse_args()
    sampling_args = {
            'discriminator' :DiscriminatorNetCIFAR10,            
            "pretrained_discriminatorPath": args.discriminator,
            'images_path': 'demo',
            "explanationTypes" : ["lime2","integrated_gradients", "saliency", "shapley_value_sampling","deeplift"],
            "image_w":32,
            "image_h":32,
        }
    run_explaination_demo(sampling_args)


def run_explaination_demo(args):
    """
    This function is supposed to calculate metrics for coco.
    :param path: path of the generator model
    :type path: str
    :param numberOfSamples: number of samples to generate
    :type numberOfSamples: int
    :return: None
    :rtype: None
    """
    folder = f'{os.getcwd()}/demo_explained'
    if not os.path.exists(folder):
        os.makedirs(folder)


    images_path=f"{os.getcwd()}/{args['images_path']}"
    image_w=args['image_w']
    image_h=args['image_h']

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.Resize((image_w,image_h))])
    data_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(images_path,transform=transform), 
                                              batch_size=1, shuffle=False)
                                              
    run_explaination_demo_on_images(data_loader,args, path_output=folder)
    return


def run_explaination_demo_on_images(loader,args,path_output):
    """
    This function runs explanations on the given images  and save the explanation results as images
    :return: None
    :rtype: None
    """
    explanation_types = args["explanationTypes"]
    discriminatorArch = args["discriminator"]
    disc_path = args['pretrained_discriminatorPath']

    # load saved weights for discriminator & intialize Models
    discriminator = discriminatorArch()
    #discriminator = nn.Dataparallel(discriminator)
    discriminator.load_state_dict(torch.load(disc_path, map_location=lambda storage, loc: storage)['model_state_dict'])
    discriminator.eval()
    for i_image,  (real_batch) in enumerate(loader):
        sample, label = real_batch    
        sample_image = sample.squeeze(0).numpy()
        sample_image = np.transpose(sample_image, (1, 2, 0))
        sample_image = ((sample_image/2) + 0.5) * 255
        sample_image = sample_image.astype(np.uint8)
            
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
            figure.savefig(f'{path_output}/{i_image}_{type}.jpg')
            
            
        sample.requires_grad = False
    return

if __name__ == "__main__":
    main()