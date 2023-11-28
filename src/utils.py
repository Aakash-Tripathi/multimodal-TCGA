import numpy as np
import pandas as pd


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
