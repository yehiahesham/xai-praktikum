import os
import wget
import pandas as pd
from transformers import CLIPProcessor, CLIPTokenizerFast, CLIPModel
import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
from tqdm.auto import tqdm
import glob
import multiprocessing

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_request(query):
    print(query.url)

def read_file_from_tsv(file_path):
    data = pd.read_csv(file_path, sep="\t")
    return data

def download_element(url, save_loc, error_list, i):
    try:
        file_name = wget.download(url, out=save_loc)
        ext_idx = file_name.rfind('.')
        if ext_idx == -1:
            error_list.append(f"URL {i} failed. Does not have extention" )
            return
        ext = file_name[ext_idx:]
        os.rename(file_name, save_loc + "/" + str(i) + ext)

        print(file_name, save_loc + "/" + str(i) + ext)

    except Exception as e:
        error_list.append(f"URL failed." + str(e))


def download_dataset(file_path, download_path, threadıng=False):

    if threadıng:
        cpus = multiprocessing.cpu_count()
        max_pool_size = 4
        pool = multiprocessing.Pool(cpus if cpus < max_pool_size else max_pool_size)

    text_url_pairs = read_file_from_tsv(file_path)
    save_loc = download_path + "/Conceptual Captions/images/"
    start_file = 0
    downloaded_files = []
    error_list = []
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    else:
        downloaded_files = len([name for name in os.listdir(save_loc) if os.path.isfile(os.path.join(save_loc, name))])
        # start_file = len([name for name in os.listdir(save_loc) if os.path.isfile(os.path.join(save_loc, name))])

    # print("Index of the starting url is: ", start_file)
    for i in tqdm(range(0, len(text_url_pairs))):

        if glob.glob(save_loc + "/" + str(i) + ".*"):
            continue

        text, url = text_url_pairs.iloc[i][0], text_url_pairs.iloc[i][1]
        download_element(url, save_loc, error_list, i)

        if threadıng:
            pool.close()
            pool.join()





download_dataset("data/Conceptual Captions/Train_GCC-training.tsv", "data", threadıng=False)


# image = Image.open(file_name)

# response = requests.get(url)
#
# image = Image.open(response.content)
#
# if response.status_code:
#     fp = open(, "wb")
#     fp.write(image)
#     fp.close()
# model_id = "openai/clip-vit-base-patch32"
#
# # Loading the CLIP models
# model = CLIPModel.from_pretrained(model_id).to(device)
# tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
# processor = CLIPProcessor.from_pretrained(model_id)
#
# # Getting the data
# data = read_file_from_tsv("data/Conceptual Captions/Validation_GCC-1.1.0-Validation.tsv")
# test_data_caption = data.iloc[1][0]
# print(test_data_caption)
#
# # Testing the text embedding
# inputs = tokenizer(test_data_caption, return_tensors="pt")
# print(inputs)
#
# # Testing the image embedding
# # print(data.info)
# urls = []
#
#
# for i in range(len(data)):
#     urls.append(data.iloc[i][1])
#
# print(urls)

# Resize the image
# image = processor(
#     text=None,
#     images= img,
#     return_tensors="pt"
# )['pixel_values'].to(device)
# print(image.shape) [1, 3, 224, 224]
#
# image = image.cpu().numpy()
# Show the resized img
# plt.imshow(image.squeeze(0).T.astype('uint8'))
# plt.show()
#
# # Get image features
# image_emb = model.get_image_features(torch.tensor(image).to(device))
# print(image_emb.shape) # 1x512


# Import a batch of images, say 100
# np.random.seed(0)
# sample_idx = np.random.randint(0, len(data) + 1, 100).tolist()
# images = [data.iloc[i][1] for i in sample_idx]
#
#
# batch_size = 16
# image_Arr = None
#
# for i in tqdm(range(0, len(images), batch_size)):
#     # Select batch of images
#     batch = images[i:i+batch_size]
#     # TODO:
#
