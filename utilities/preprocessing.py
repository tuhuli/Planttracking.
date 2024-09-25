from typing import List

import pandas as pd
from utilities.trackedObject import TrackedObject


def get_one_object(data, objects_ids: List[int], start=False):
    return data[data['Object_ID'].isin(objects_ids)]


def parse_ground_truth(path):
    columns = ['Frame', 'Object_ID', 'Centre_X', 'Centre_Y', 'Half_X', 'Half_Y', 'Confidence', 'Object_type',
               'Visibility']

    data = pd.read_csv(path + '/gt/gt.txt', names=columns, delimiter=',', header=None)

    data = centre_data(data)

    return data


def parse_detections(path):
    columns = ['Frame', 'Object_ID', 'X1', 'Y1', 'X2', 'Y2', 'Confidence']

    data = pd.read_csv(path + '/det/det.txt', names=columns, delimiter=',', header=None)

    data = centre_data(data)

    return data


def centre_data(data):
    data["Half_X"] = data["Half_X"] / 2
    data["Half_Y"] = data["Half_Y"] / 2

    data["Centre_X"] = data["Centre_X"] + data["Half_X"]
    data["Centre_Y"] = data["Centre_Y"] + data["Half_Y"]
    return data


def initialize_tr(data):
    TR_objects = []
    for index, row in data.iterrows():
        tr = TrackedObject(row['Centre_X'], row['Centre_Y'], row["Half_X"], row["Half_Y"])
        TR_objects.append(tr)
    return TR_objects
