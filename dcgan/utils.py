import pandas as pd
from transformers import CLIPProcessor, CLIPTokenizerFast, CLIPModel
import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt
from urllib.request import urlopen
import numpy as np
from tqdm.auto import tqdm

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_request(query):
    print(query.url)

def read_file_from_tsv(file_path):
    data = pd.read_csv(file_path, sep="\t")
    return data


model_id = "openai/clip-vit-base-patch32"

# Loading the CLIP models
model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# Getting the data
data = read_file_from_tsv("data/Conceptual Captions/Train_GCC-training.tsv")
test_data_caption = data.iloc[0][0]
print(test_data_caption)

# Testing the text embedding
inputs = tokenizer(test_data_caption, return_tensors="pt")
print(inputs)

# Testing the image embedding
print(data.info)
urls = []


for i in range(len(data)):
    urls.append(data.iloc[i][1])

print(urls)

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
