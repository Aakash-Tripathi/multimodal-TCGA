import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from lib.MINDS import MINDS
from pathlib import Path
from tqdm import tqdm
import json
import os
import torch
from src.models.modality import ClinincalEmbeddings, REMEDISpathology, REMEDISradiology


def generate_pathology_embeddings_shapes_csv(filename):
    with open("/mnt/d/TCGA-LUAD/manifest.json", "r") as file:
        manifest = json.load(file)
    results = []
    for item in tqdm(manifest, desc="Checking for pathology embeddings"):
        case_id = item["case_id"]
        if "Slide Image" not in item:
            continue
        embedding_path = os.path.join(
            "/mnt/w/data/TCGA-LUAD/embeddings", case_id, "pathology.npy"
        )
        try:
            path_embedding = np.load(embedding_path)
            path_embedding_shape = path_embedding.shape
        except FileNotFoundError:
            path_embedding_shape = "File not found"
        except Exception as e:
            path_embedding_shape = str(e)
        results.append(f"{case_id},{path_embedding_shape}")
    with open(filename, "w") as file:
        file.write("\n".join(results))


class InitialSetup:
    def __init__(self, project_id, embeddings_dir, raw_data_dir):
        self.project_id = project_id
        self.embeddings_dir = embeddings_dir
        self.raw_data_dir = raw_data_dir
        self.preprocess_dir = embeddings_dir + "/../preprocessing/"

    def generate_cohort_data(self):
        minds = MINDS()
        filename = "/mnt/w/data/" + self.project_id
        if Path(f"{filename}.json").exists():
            json_objects = json.load(open(f"{filename}.json"))
        else:
            json_objects = {}
            tables = minds.get_tables()
            for table in tqdm(tables["Tables_in_nihnci"]):
                query = (
                    f"SELECT * FROM nihnci.{table} WHERE project_id='{self.project_id}'"
                )
                df = minds.query(query)
                for case_id, group in tqdm(
                    df.groupby("case_submitter_id"), leave=False
                ):
                    if case_id not in json_objects:
                        json_objects[case_id] = {}
                    common_fields, nested_objects = process_group(group)
                    json_objects[case_id].update(common_fields)
                    json_objects[case_id][table] = nested_objects
            with open(f"{filename}.json", "w") as fp:
                json.dump(json_objects, fp, indent=4, default=convert_numpy)
        return json_objects

    def generate_clinical_embeddings(self):
        #  Setting up the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        with open("/mnt/w/data/TCGA-LUAD.json", "r") as file:
            json_objects = json.load(file)
        clinical_model = ClinincalEmbeddings(
            model="UFNLP/gatortron-base", device=device
        )
        for case_id, patient_data in tqdm(json_objects.items()):
            if os.path.exists(self.embeddings_dir + f"/{case_id}/ehr.npy"):
                continue
            summary = generate_summary_from_json(patient_data)
            embedding = clinical_model.generate_embeddings(summary)
            embedding = embedding.detach().cpu().numpy()
            if not os.path.exists(self.embeddings_dir + f"/{case_id}"):
                os.makedirs(self.embeddings_dir + f"/{case_id}")
            np.save(self.embeddings_dir + f"/{case_id}/ehr.npy", embedding)

    def generate_pathology_embeddings(self):
        pathology_model = REMEDISpathology(
            remedis_model_path="/mnt/d/Models/REMEDIS/Pretrained-Weights/path-50x1-remedis-m"
        )

        # load the manifest.json file in the raw_data_dir
        with open(self.raw_data_dir + "/manifest.json", "r") as file:
            manifest = json.load(file)

        for item in tqdm(manifest, desc="Generating patch coordinates"):
            case_id = item["case_id"]
            preprocess_dir = self.preprocess_dir + f"/{case_id}"
            if (
                os.path.exists(self.embeddings_dir + f"/{case_id}/pathology.npy")
                or "Slide Image" not in item
            ):
                continue
            # for the len of the Slide Image list, try to generate the patches, stop when successful or when the list is exhausted
            for slide_image in item["Slide Image"]:
                slide_image_path = (
                    self.raw_data_dir + f"/raw/{case_id}/Slide Image/{slide_image}/"
                )
                patch_save_dir = pathology_model.generate_h5(
                    svs_file_path=slide_image_path, preprocess_dir=preprocess_dir
                )
                files_in_patch_dir = os.listdir(patch_save_dir)
                if len(files_in_patch_dir) > 0:
                    break
            if len(files_in_patch_dir) == 0:
                print("No patches found for:", case_id)
                continue

        for item in tqdm(manifest, desc="Generating Embeddings"):
            case_id = item["case_id"]
            if "Slide Image" not in item:
                continue
            slide_image = item["Slide Image"][0]
            svs_file_path = (
                self.raw_data_dir + f"/raw/{case_id}/Slide Image/{slide_image}/"
            )
            svs_file_path = svs_file_path + os.listdir(svs_file_path)[0]
            h5_file_path = self.preprocess_dir + f"/{case_id}/patches/"
            # check to see if the file exists, if not then continue
            if not os.path.exists(h5_file_path):
                continue
            try:
                h5_file_path = h5_file_path + os.listdir(h5_file_path)[0]
            except:
                print("No h5 file found for:", case_id)

            if os.path.exists(self.embeddings_dir + f"/{case_id}/pathology.npy"):
                continue
            embedding = pathology_model.generate_embeddings(svs_file_path, h5_file_path)
            if not os.path.exists(self.embeddings_dir + f"/{case_id}"):
                os.makedirs(self.embeddings_dir + f"/{case_id}")
            np.save(self.embeddings_dir + f"/{case_id}/pathology.npy", embedding)

    def generate_radiology_embeddings(self):
        # generate random embeddings for radiology
        with open("/mnt/w/data/TCGA-LUAD.json", "r") as file:
            json_objects = json.load(file)
        for case_id, patient_data in tqdm(
            json_objects.items(), desc="Generating RANDOM radiology embeddings"
        ):
            # random number of patches = 1 - 224
            if os.path.exists(self.embeddings_dir + f"/{case_id}/radiology.npy"):
                continue
            random_number_of_patches = np.random.randint(1, 224)
            embedding = np.random.rand(random_number_of_patches, 7, 7, 2048)
            np.save(self.embeddings_dir + f"/{case_id}/radiology.npy", embedding)

    def generate_all_embeddings(self):
        self.generate_clinical_embeddings()
        self.generate_pathology_embeddings()
        self.generate_radiology_embeddings()


def calculate_survival_time(patient_data):
    survival_times = {}

    #  year_of_diagnosis, year_of_birth, vital_status, year_of_death, age_at_diagnosis, age_at_index
    total_survival_time, count = 0, 0
    for patient_id, data in patient_data.items():
        if data["vital_status"] == "Alive":
            if data["year_of_diagnosis"]:
                survival_time = datetime.now().year - data["year_of_diagnosis"]
        elif data["vital_status"] == "Dead":
            if data["year_of_birth"] and data["age_at_diagnosis"]:
                estimated_death_year = data["year_of_birth"] + (
                    data["age_at_diagnosis"] / 365.25
                )
                if data["year_of_diagnosis"]:
                    survival_time = estimated_death_year - data["year_of_diagnosis"]
                else:
                    survival_time = None
            else:
                survival_time = None
        else:
            survival_time = None

        survival_times[patient_id] = survival_time

        if survival_time is not None:
            total_survival_time += survival_time
            count += 1

    # Impute missing survival times with the average survival time
    average_survival_time = total_survival_time / count if count > 0 else None
    for patient_id in survival_times:
        if survival_times[patient_id] is None and average_survival_time is not None:
            survival_times[patient_id] = average_survival_time

    return survival_times


def normalize_embedding(embedding, output_size):
    """
    Normalize the size of the embedding to a fixed size using adaptive pooling.
    :param embedding: Input embedding tensor.
    :param output_size: The target output size (H, W).
    :return: Normalized embedding tensor.
    """
    adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
    return adaptive_pool(embedding)


def freeze_model(model):
    """
    Freeze the parameters of a model.
    :param model: The model to be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False


def generate_summary_from_json(patient_data):
    # Initialize an empty list to store sentences
    summary_sentences = []

    # Iterate through each key-value pair in the JSON object
    for key, value in patient_data.items():
        # if the key is "case_id" then skip it
        if key == "case_id" or key == "pathology_report_uuid":
            continue

        # remove all _ from the key
        key = key.replace("_", " ")
        sentence = f"{key}: {value};"

        # if the value is a list, then skip it
        if isinstance(value, list):
            continue

        summary_sentences.append(sentence)

    # Compile all sentences into a single summary string
    summary = " ".join(summary_sentences)

    return summary


def process_group(group):
    common_fields = {}
    nested_objects = []
    for col in group.columns:
        unique_values = group[col].dropna().unique()
        if len(unique_values) == 1:
            # If only one unique value exists, it's a common field
            common_fields[col] = unique_values[0]

    # Create nested objects for fields that are not common
    for idx, row in group.iterrows():
        nested_object = {
            col: row[col]
            for col in group.columns
            if col not in common_fields and pd.notna(row[col])
        }
        if nested_object:  # Only add if the nested object is not empty
            nested_objects.append(nested_object)

    return common_fields, nested_objects


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def flatten_json(y):
    out = {}

    def flatten(x, name=""):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + "_")
        elif type(x) is list:
            # ignore the list for now
            pass
        else:
            out[name[:-1]] = x

    flatten(y)
    return out
