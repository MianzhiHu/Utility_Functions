import numpy as np
import pandas as pd
import ast
import os
import json


def process_participant_data(participant_path, sgt_pos=1, igt_pos=3, img_pos=2):
    dfs = []
    # iterate each sub‑folder
    for folder_name in os.listdir(participant_path):
        folder_path = os.path.join(participant_path, folder_name)

        # load each .txt as JSON lines
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        # skip empty lines
                        continue
                    try:
                        dfs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # remove the empty dfs
    dfs = [df for df in dfs if df]

    # extract demo info
    demo_info = {k: ast.literal_eval(v) for k, v in dfs[0].items()}
    demo_info = pd.DataFrame(demo_info).reset_index(drop=True)

    # change all the dfs to DataFrame
    for i in [igt_pos, img_pos, sgt_pos]:
        dfs[i] = pd.DataFrame(dfs[i])

    dfs[sgt_pos]['Task'] = 'SGT'
    dfs[igt_pos]['Task'] = 'IGT'
    dfs[img_pos]['Task'] = 'ImageRating'

    # combine all the dfs into one
    df = pd.concat([demo_info, dfs[sgt_pos], dfs[img_pos], dfs[igt_pos]], ignore_index=True)
    df[['Gender', 'Ethnicity', 'Race', 'Age']] = df[['Gender', 'Ethnicity', 'Race', 'Age']].ffill()
    df = df.iloc[1:]
    return df


def determine_condition(group):
    image_names = group['image_name'].astype(str)
    if image_names.str.contains('Nature').any():
        return 'Nature'
    elif image_names.str.contains('Urban').any():
        return 'Urban'
    elif image_names.str.contains('edge').any():
        return 'Control'
    return 'unknown'