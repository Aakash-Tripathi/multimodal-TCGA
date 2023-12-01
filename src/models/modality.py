import os
import h5py
import slideio
import cv2
from tqdm import tqdm
import numpy as np
import gc
import concurrent.futures
from src.models.create_patches_fp import seg_and_patch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ClinincalEmbeddings:
    def __init__(self, model, device):
        from transformers import AutoModel, AutoTokenizer, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.device = device
        self.model = self.model.to(self.device)

    def generate_embeddings(self, sentences):
        import torch

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
    def __init__(self, remedis_model_path):
        self.remedis_model_path = remedis_model_path

    def generate_h5(self, svs_file_path, preprocess_dir):
        source = svs_file_path  # path to folder containing raw wsi image files
        save_dir = preprocess_dir  # path to folder to save process_list.csv
        patch_save_dir = os.path.join(save_dir, "patches")
        mask_save_dir = os.path.join(save_dir, "masks")
        stitch_save_dir = os.path.join(save_dir, "stitches")

        patch_size = 512
        step_size = 512

        seg = True
        stitch = False
        patch = True
        no_auto_skip = True
        patch_level = 0

        directories = {
            "source": source,
            "save_dir": save_dir,
            "patch_save_dir": patch_save_dir,
            "mask_save_dir": mask_save_dir,
            "stitch_save_dir": stitch_save_dir,
        }

        for key, val in directories.items():
            if key not in ["source"]:
                os.makedirs(val, exist_ok=True)

        seg_params = {
            "seg_level": -1,
            "sthresh": 8,
            "mthresh": 7,
            "close": 4,
            "use_otsu": False,
            "keep_ids": "none",
            "exclude_ids": "none",
        }
        filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
        vis_params = {"vis_level": -1, "line_thickness": 250}
        patch_params = {"use_padding": True, "contour_fn": "four_pt"}

        parameters = {
            "seg_params": seg_params,
            "filter_params": filter_params,
            "patch_params": patch_params,
            "vis_params": vis_params,
        }

        seg_times, patch_times = seg_and_patch(
            **directories,
            **parameters,
            patch_size=patch_size,
            step_size=step_size,
            seg=seg,
            use_default_params=False,
            save_mask=True,
            stitch=stitch,
            patch_level=patch_level,
            patch=patch,
            auto_skip=no_auto_skip,
        )

        return patch_save_dir

    def load_image(self, svs_file_path, h5_file_path):
        file = h5py.File(h5_file_path, "r")
        coords = file["coords"]
        slide = slideio.open_slide(svs_file_path, "SVS")
        return slide, coords

    def process_patch(self, slide, coord, patch_shape, resize_shape):
        import tensorflow as tf

        x, y = coord
        scene = slide.get_scene(0)
        image = scene.read_block((x, y, patch_shape, patch_shape))
        image = cv2.resize(image, resize_shape)
        image = tf.cast(image, tf.float32)
        return image.numpy()  # Convert to numpy array immediately to save memory

    def get_patches(self, slide, coords):
        patch_shape = 512
        resize_shape = (224, 224)

        def process_in_batches(coords_batch):
            patches_batch = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        self.process_patch, slide, coord, patch_shape, resize_shape
                    )
                    for coord in coords_batch
                ]
                for future in concurrent.futures.as_completed(futures):
                    patches_batch.append(future.result())
            return np.array(patches_batch)

        # Process in batches
        batch_size = 100  # Adjust batch size based on memory constraints
        patches = []
        for i in tqdm(
            range(0, len(coords), batch_size),
            desc="Processing patch batches",
            leave=False,
        ):
            coords_batch = coords[i : i + batch_size]
            patches_batch = process_in_batches(coords_batch)
            patches.append(patches_batch)

        patches = np.concatenate(patches, axis=0)
        patches = patches.reshape(-1, *resize_shape, 3)
        return patches

    def generate_embeddings(self, svs_file_path, h5_file_path):
        import tensorflow as tf

        slide, coords = self.load_image(svs_file_path, h5_file_path)
        patches = self.get_patches(slide, coords)
        max_patches = 24_000
        patches = patches[:max_patches]

        print("patches shape", patches.shape)

        batch_size = 100  # Adjust based on memory constraints
        pathology_model = tf.saved_model.load(self.remedis_model_path)

        embeddings = None
        for i in tqdm(
            range(0, len(patches), batch_size),
            desc="Processing embedding batches",
            leave=False,
        ):
            batch = patches[i : i + batch_size]
            with tf.device("/GPU:0"):
                batch_embeddings = pathology_model(batch).numpy()

            if embeddings is None:
                embeddings = batch_embeddings
            else:
                embeddings = np.concatenate((embeddings, batch_embeddings), axis=0)

            # Clear memory
            del batch, batch_embeddings
            tf.keras.backend.clear_session()
            gc.collect()

        # Clear memory
        del patches, slide, coords, pathology_model
        tf.keras.backend.clear_session()
        gc.collect()

        return embeddings


class REMEDISradiology:
    def __init__(self):
        pass
