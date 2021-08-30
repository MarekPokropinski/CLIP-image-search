import torch
import clip
from PIL import Image
import os
import numpy as np
import pickle
import io
import argparse
import tqdm
import PySimpleGUI as sg
from sklearn.neighbors import NearestNeighbors


if os.path.exists('./index.pickle'):
    with open('./index.pickle', 'rb') as f:
        index = pickle.load(f)

else:
    print("Didn't find index.pickle! Generate index first!")
    exit(-1)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

X = list(index.values())
y = list(index.keys())

y = np.array(y)

nn = NearestNeighbors(metric='cosine')
nn.fit(X, y)

file_list_column = [
    [
        sg.In(size=(30, 1), enable_events=True, key="-SEARCH-"),
        sg.Button("Search", key="-SEARCH BUTTON-"),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]


image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(50, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]


layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image search", layout)

def show_image(path):
    window["-TOUT-"].update(os.path.basename(path))
    if os.path.exists(path):
        image = Image.open(path)
        image.thumbnail((400, 400))
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        window["-IMAGE-"].update(data=bio.getvalue())

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = values["-FILE LIST-"][0]
            show_image(filename)
        except:
            pass

    elif event == "-SEARCH BUTTON-":
        query = values["-SEARCH-"]
        text = clip.tokenize(query).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text).cpu().numpy()
        [dist], [ind] = nn.kneighbors(text_features, n_neighbors=100, return_distance=True)
        search_results = y[ind]
        window["-FILE LIST-"].update(search_results)
        show_image(search_results[0])


window.close()
