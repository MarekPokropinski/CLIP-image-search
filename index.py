import torch
import clip
from PIL import Image
import os
import numpy as  np
import pickle
import argparse
import tqdm

BATCH_SIZE = 64

image_extensions = ['.png', '.jpg']

parser = argparse.ArgumentParser(description='Index images for image search.')
parser.add_argument('path', metavar='path', type=str, help='path with images to index')

path = parser.parse_args().path

image_paths = []

for file in os.listdir(path):
    filepath = os.path.join(path, file)
    if not os.path.isfile(filepath):
        continue
    for suffix in image_extensions:
        if file.endswith(suffix):
            image_paths.append(filepath)
            break

# print('Files to process: \n', '\n'.join(image_paths))
# exit()


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


indexes = {}
if os.path.exists('./index.pickle'):
    with open('./index.pickle', 'rb') as f:
        indexes = pickle.load(f)

for i in tqdm.tqdm(range(0, len(image_paths), BATCH_SIZE)):
    mb_paths = image_paths[i:i+BATCH_SIZE]
    images = []
    corrupted_images = []
    for path in mb_paths:
        try:
            image = preprocess(Image.open(path))
            images.append(image)
        except Exception as e:
            print('Could not process image:', path, ', error:', e)
            corrupted_images.append(path)

    for corrupted_img in corrupted_images:
        mb_paths.remove(corrupted_img)

    images = torch.stack(images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images).cpu().numpy()

    
    for j, path in enumerate(mb_paths):
        indexes[path] = image_features[j].tolist()


with open('./index.pickle', 'wb') as f:
    pickle.dump(indexes, f)