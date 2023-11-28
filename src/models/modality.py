from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import tensorflow_hub as hub
import os
import h5py
import slideio
import cv2
from tqdm import tqdm
import tensorflow as tf
import numpy as np


class ClinincalEmbeddings:
    def __init__(self, model, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.device = device
        self.model = self.model.to(self.device)

    def generate_embeddings(self, sentences):
        inputs = self.tokenizer(
            str(sentences),
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        # embedding = outputs["last_hidden_state"].cpu().numpy()
        # take the mean
        embedding = torch.mean(outputs["last_hidden_state"], dim=1)
        return embedding


class REMEDISpathology:
    def __init__(self, svs_file_path, h5_file_path):
        remedis_model_path = (
            os.getcwd() + "/models/REMEDIS/Pretrained-Weights/path-50x1-remedis-m"
        )
        self.pathology_model = hub.load(remedis_model_path)
        self.svs_file_path = svs_file_path
        self.h5_file_path = h5_file_path

    def load_image(self):
        h5_file = h5py.File(self.h5_file_path, "r")
        coords = h5_file["coords"]
        slide = slideio.open_slide(self.svs_file_path, "SVS")
        return slide, coords

    def generate_patches(self, slide, coords):
        patch_shape = coords[1][1] - coords[0][1]
        patches = []
        for coord in tqdm(coords):
            x, y = coord
            scene = slide.get_scene(0)
            image = scene.read_block((x, y, patch_shape, patch_shape))
            image = cv2.resize(image, (224, 224))
            image = tf.cast(image, tf.float32)
            image = tf.expand_dims(image, axis=0)
            patches.append(image)
        patches = np.array(patches)
        patches = patches.reshape(-1, 224, 224, 3)
        return patches

    def generate_embeddings(self, patches):
        embeddings = self.pathology_model(patches)
        return embeddings


class REMEDISradiology:
    def __init__(self):
        pass
