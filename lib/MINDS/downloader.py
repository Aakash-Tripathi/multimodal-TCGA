import os
import re
import requests
import json
import tarfile
import shutil
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from google.cloud import storage
from urllib.parse import urlparse
from retry import retry
from itertools import chain
from functools import partial


class GDCFileDownloader:
    """
    Class for downloading files from the GDC API based on case_ids.
    """

    def __init__(self, DATA_DIR):
        """
        Initialize the downloader with a specific data directory.

        :param DATA_DIR: Directory where downloaded data will be stored.
        """
        self.BASE_URL = "https://api.gdc.cancer.gov/"
        self.FILES_ENDPOINT = "files"
        self.DATA_ENDPOINT = "data"
        self.DATA_DIR = DATA_DIR

    def get_file_uuids_for_case_id(self, case_id):
        """
        Fetch file UUIDs from the GDC API based on a given case_id.

        :param case_id: The ID of the case to fetch file UUIDs for.
        :return: List of file UUIDs associated with the given case_id.
        """
        params = {
            "filters": json.dumps(
                {
                    "op": "and",
                    "content": [
                        {
                            "op": "=",
                            "content": {"field": "cases.case_id", "value": [case_id]},
                        },
                        {"op": "=", "content": {"field": "access", "value": ["open"]}},
                    ],
                }
            ),
            "fields": "file_id",
            "format": "JSON",
            "size": "1_000_000",
        }
        response = requests.get(self.BASE_URL + self.FILES_ENDPOINT, params=params)
        return [entry["file_id"] for entry in response.json()["data"]["hits"]]

    def download_files_for_case_id(self, case_id):
        """
        Download all files associated with a given case_id.

        :param case_id: The ID of the case to download files for.
        """
        try:
            file_uuid_list = self.get_file_uuids_for_case_id(case_id)
            response = requests.post(
                self.BASE_URL + self.DATA_ENDPOINT,
                data=json.dumps({"ids": file_uuid_list}),
                headers={"Content-Type": "application/json"},
            )

            if "Content-Disposition" in response.headers:
                file_name = re.findall(
                    "filename=(.+)", response.headers["Content-Disposition"]
                )[0]
                file_extension = file_name.split(".")[-1]
                os.makedirs(self.DATA_DIR, exist_ok=True)
                output_path = os.path.join(self.DATA_DIR, f"{case_id}.{file_extension}")
                with open(output_path, "wb") as output_file:
                    output_file.write(response.content)
            else:
                print("Content-Disposition header missing")

        except Exception as e:
            print(f"Failed to download: {case_id}")

    def extract_files(self, ext, mode):
        for filename in os.listdir(self.DATA_DIR):
            if filename.endswith(ext):
                filepath = os.path.join(self.DATA_DIR, filename)
                case_id = filename.replace(".tar.gz", "")
                extraction_successful = False

                while not extraction_successful:
                    try:
                        with tarfile.open(filepath, mode) as tar:
                            tar.extractall(path=self.DATA_DIR)
                            extraction_successful = True
                    except EOFError as e:
                        print(
                            f"Extraction failed due to corrupted file: {e}. Deleting and re-downloading."
                        )
                        os.remove(filepath)
                        self.download_files_for_case_id(case_id)
                    except PermissionError as e:
                        print(f"Permission Error: {e}. Retrying extraction.")
                        continue
                    except Exception as e:
                        print(
                            f"An unexpected error occurred: {e}. Deleting and re-downloading."
                        )
                        os.remove(filepath)
                        self.download_files_for_case_id(case_id)

    def organize_files(self, case_id):
        """
        Organize files in the data directory into subdirectories by case_id and data type.

        :param case_id: The ID of the case to organize files for.
        """
        target_dir = os.path.join(self.DATA_DIR, "raw", case_id)
        for file_uuid in self.get_file_uuids_for_case_id(case_id):
            response = requests.get(
                self.BASE_URL + self.FILES_ENDPOINT + "/" + file_uuid
            )
            if response.status_code == 200:
                data_type = response.json()["data"]["data_type"]
                os.makedirs(os.path.join(target_dir, data_type), exist_ok=True)
                try:
                    shutil.move(
                        os.path.join(self.DATA_DIR, file_uuid),
                        os.path.join(target_dir, data_type, file_uuid),
                    )
                except (FileNotFoundError, FileExistsError, shutil.Error):
                    pass
            elif response.status_code == 429:
                time.sleep(30)
                self.organize_files(case_id)

    def generate_manifest(self):
        """
        Generate a manifest.json file in the data directory that logs all files in the /raw subdirectory.
        """
        manifest = []
        raw_dir = os.path.join(self.DATA_DIR, "raw")
        for case_id in os.listdir(raw_dir):
            case_dir = os.path.join(raw_dir, case_id)
            if not os.path.isdir(case_dir):
                continue
            case_manifest = {"case_id": case_id}
            for data_type in os.listdir(case_dir):
                data_type_dir = os.path.join(case_dir, data_type)
                data_manifest = []
                for filename in os.listdir(data_type_dir):
                    file_uuid = os.path.splitext(filename)[0]
                    data_manifest.append(file_uuid)
                case_manifest[data_type] = data_manifest
            manifest.append(case_manifest)
        with open(os.path.join(self.DATA_DIR, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=4)

    def post_process_cleanup(self):
        """
        Clean up the data directory by removing all .gz, .tar, and .txt files.
        """
        for filename in tqdm(os.listdir(self.DATA_DIR)):
            if (
                filename.endswith(".gz")
                or filename.endswith(".tar")
                or filename.endswith(".txt")
            ):
                filepath = os.path.join(self.DATA_DIR, filename)
                os.remove(filepath)

    def rename(self, case_ids, case_submitter_ids):
        raw_data_path = os.path.join(self.DATA_DIR, "raw")

        # Create a mapping of case_ids to their corresponding case_submitter_ids
        case_mapping = dict(zip(case_ids, [str(x[0]) for x in case_submitter_ids]))

        # Renaming directories
        for case_id, case_submitter_id in tqdm(case_mapping.items()):
            case_id_path = os.path.join(raw_data_path, case_id)
            case_submitter_id_path = os.path.join(raw_data_path, case_submitter_id)

            if os.path.exists(case_id_path):
                os.rename(case_id_path, case_submitter_id_path)

        # Reading manifest.json
        manifest_path = os.path.join(self.DATA_DIR, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)

        # Updating case_id in manifest.json
        for item in manifest_data:
            old_case_id = item["case_id"]
            if old_case_id in case_mapping:
                item["case_id"] = case_mapping[old_case_id]

        # Writing updated manifest.json
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)

    def multi_download(self, case_ids):
        """
        Concurrently download files for multiple case_ids.

        :param case_ids: List of case IDs to download files for.
        """
        thread_map(self.download_files_for_case_id, case_ids)

    def multi_extract(self):
        """
        Concurrently extract all .gz and .tar files in the data directory.
        """
        thread_map(
            lambda ext, mode: self.extract_files(ext, mode),
            [".gz", ".tar"],
            ["r:gz", "r"],
        )

    def multi_organize(self, case_ids):
        """
        Concurrently organize files for multiple case_ids.

        :param case_ids: List of case IDs to organize files for.
        """
        thread_map(self.organize_files, case_ids)

    def process_cases(self, case_ids, case_submitter_ids):
        """
        Process a list of case_ids by downloading, extracting, organizing, and cleaning up files.

        :param case_ids: List of case IDs to process.
        """
        self.multi_download(case_ids)
        self.multi_extract()
        self.multi_organize(case_ids)
        self.post_process_cleanup()
        self.generate_manifest()
        self.rename(case_ids, case_submitter_ids)


class IDCFileDownloader:
    def __init__(self, save_directory):
        self.idc_api_preamble = "https://api.imaging.datacommons.cancer.gov/v1"
        self.save_directory = save_directory

    @retry(tries=5, delay=2, backoff=2)
    def make_api_call(self, url, params, body):
        response = requests.post(url, params=params, json=body)
        if response.status_code == 200:
            data = response.json()
            if "manifest" not in data:
                return data
            totalFound = data["manifest"]["totalFound"]
            rowsReturned = data["manifest"]["rowsReturned"]
            if totalFound <= rowsReturned:
                return data
            else:
                params["page_size"] = totalFound + 10
                return self.make_api_call(url, params, body)
        else:
            raise Exception(f"Request failed: {response.reason}")

    def get_manifest_preview(self, filters):
        url = f"{self.idc_api_preamble}/cohorts/manifest/preview"
        params = dict(
            sql=False,
            Collection_ID=True,
            Patient_ID=True,
            StudyInstanceUID=True,
            SeriesInstanceUID=True,
            SOPInstanceUID=True,
            Source_DOI=True,
            CRDC_Study_GUID=True,
            CRDC_Series_GUID=True,
            CRDC_Instance_GUID=True,
            GCS_URL=True,
        )
        body = {
            "name": "MINDS",
            "description": "MINDS",
            "filters": filters,
        }
        return self.make_api_call(url, params, body)

    def get_query_preview(self, filters):
        url = f"{self.idc_api_preamble}/cohorts/query/preview"
        params = dict(
            sql=False,
        )
        cohort_def = {
            "name": "testcohort",
            "description": "Test description",
            "filters": filters,
        }
        queryFields = {
            "fields": [
                "Modality",
                "BodyPartExamined",
                "StudyDescription",
                "StudyInstanceUID",
                "PatientID",
                "Program",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "SeriesDescription",
                "SliceThickness",
                "SeriesNumber",
                "StudyDate",
                "SOPClassUID",
                "collection_id",
                "AnatomicRegionSequence",
                "SegmentedPropertyCategoryCodeSequence",
                "SegmentedPropertyTypeCodeSequence",
                "FrameOfReferenceUID",
                "SegmentNumber",
                "SegmentAlgorithmType",
                "SUVbw",
                "Volume",
                "Diameter",
                "Surface_area_of",
                "Total_Lesion_Glycolysis",
                "Standardized_Added_Metabolic_Activity",
                "Percent_Within_First_Quarter_of_Intensity_Range",
                "Percent_Within_Third_Quarter_of_Intensity_Range",
                "Percent_Within_Fourth_Quarter_of_Intensity_Range",
                "Percent_Within_Second_Quarter_of_Intensity_Range",
                "Standardized_Added_Metabolic_Activity_Background",
                "Glycolysis_Within_First_Quarter_of_Intensity_Range",
                "Glycolysis_Within_Third_Quarter_of_Intensity_Range",
                "Glycolysis_Within_Fourth_Quarter_of_Intensity_Range",
                "Glycolysis_Within_Second_Quarter_of_Intensity_Range",
                "Internal_structure",
                "Sphericity",
                "Calcification",
                "Lobular_Pattern",
                "Spiculation",
                "Margin",
                "Texture",
                "Subtlety_score",
                "Malignancy",
                "Apparent_Diffusion_Coefficient",
                "tcia_species",
                "Manufacturer",
                "ManufacturerModelName",
                "license_short_name",
            ]
        }
        body = {
            "cohort_def": cohort_def,
            "queryFields": queryFields,
        }
        return self.make_api_call(url, params, body)

    def merge_data(self, manifest_data, query_data):
        merged_data = []

        # Create dictionaries to hold lists of entries for each Patient_ID
        manifest_dict = {}
        query_dict = {}

        # Populate manifest_dict
        for manifest_entry in manifest_data.get("manifest", {}).get(
            "json_manifest", []
        ):
            patient_id = manifest_entry.get("Patient_ID")
            if patient_id not in manifest_dict:
                manifest_dict[patient_id] = []
            manifest_dict[patient_id].append(manifest_entry)

        # Populate query_dict
        for query_entry in query_data.get("query_results", {}).get("json", []):
            patient_id = query_entry.get("PatientID")
            if patient_id not in query_dict:
                query_dict[patient_id] = []
            query_dict[patient_id].append(query_entry)

        # Merge the data
        for patient_id, manifest_entries in manifest_dict.items():
            if patient_id in query_dict:
                query_entries = query_dict[patient_id]
                for manifest_entry in manifest_entries:
                    for query_entry in query_entries:
                        merged_entry = {
                            "Patient_ID": patient_id,
                            "GCS_URL": manifest_entry.get("GCS_URL"),
                            "Modality": query_entry.get("Modality"),
                        }
                        if merged_entry not in merged_data:
                            merged_data.append(merged_entry)
        return merged_data

    def download_dicom_files(self, merged_data):
        client = storage.Client.create_anonymous_client()
        aggregated_data = {}

        # Step 1: Aggregate all patient data
        for patient_data in tqdm(merged_data, desc="Aggregating data"):
            patient_id = patient_data["Patient_ID"]
            gcs_url = patient_data["GCS_URL"]
            modality = patient_data["Modality"]
            # Skip over "SM" modalities
            if modality == "SM":
                continue
            file_info = {"gcs_url": gcs_url, "modality": modality}

            if patient_id in aggregated_data:
                aggregated_data[patient_id].append(file_info)
            else:
                aggregated_data[patient_id] = [file_info]

        # Step 2: Download DICOM files
        def download_single_file(file_info, patient_id):
            gcs_url = file_info["gcs_url"]
            modality = file_info["modality"]

            parsed_url = urlparse(gcs_url)
            bucket_name = parsed_url.netloc
            blob_name = parsed_url.path.lstrip("/")
            save_path = os.path.join(
                self.save_directory, "raw", patient_id, modality, blob_name
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(save_path)

        for patient_id, file_infos in tqdm(
            aggregated_data.items(), desc="Downloading data"
        ):
            thread_map(
                lambda file_info: download_single_file(file_info, patient_id),
                file_infos,
            )

    def update_manifest(self, merged_data):
        manifest_path = os.path.join(self.save_directory, "manifest.json")

        # Read existing manifest
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f) if os.path.exists(manifest_path) else []

        # Create a dictionary for faster look-up
        manifest_dict = {item.get("case_id"): item for item in manifest_data}

        # Function to update a single manifest entry
        def update_single_entry(patient_data):
            patient_id = patient_data["Patient_ID"]
            modality = patient_data["Modality"]
            gcs_url = patient_data["GCS_URL"]
            folder_name = urlparse(gcs_url).path.strip("/").split("/")[-2]

            if modality == "SM":
                return

            if patient_id in manifest_dict:
                manifest_entry = manifest_dict[patient_id]
                if modality not in manifest_entry:
                    manifest_entry[modality] = []
                if folder_name not in manifest_entry[modality]:
                    manifest_entry[modality].append(folder_name)
            else:
                new_entry = {"case_id": patient_id, modality: [folder_name]}
                manifest_data.append(new_entry)
                manifest_dict[patient_id] = new_entry

        thread_map(update_single_entry, tqdm(merged_data, desc="Updating manifest"))

        # Save the updated manifest back to disk
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)

    def generate_merged_data_for_case(self, single_case_submitter_id):
        filters = {"PatientID": [single_case_submitter_id[0]]}
        manifest_data = self.get_manifest_preview(filters)
        query_data = self.get_query_preview(filters)

        if manifest_data and query_data:
            return self.merge_data(manifest_data, query_data)
        return None

    def process_cases(self, case_submitter_ids):
        # Generate merged_data for all cases
        all_merged_data = thread_map(
            self.generate_merged_data_for_case, case_submitter_ids
        )

        all_merged_data = list(chain.from_iterable(all_merged_data))

        self.download_dicom_files(all_merged_data)
        self.update_manifest(all_merged_data)
