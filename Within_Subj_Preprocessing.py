import re
import pandas as pd
import os
import ast
import numpy as np


def preprocess_data(main_folder_directory, behavioral_list, other_data_list,
                    numeric_list, dict_list, baseline_pos=0, compare_pos=11, compare_condition='Frequency',
                    estimate=False, cutoff=9999):
    """
    This function preprocesses data from JATOS and returns a DataFrame with the relevant columns.
    :param main_folder_directory: str, the path to the main folder containing the data
    :param behavioral_list: list, the list of columns that contain behavioral data
    :param other_data_list: list, the list of columns that contain other data
    :param numeric_list: list, the list of columns that contain numeric data
    :param dict_list: list, the list of columns that contain dictionary-like strings
    :param baseline_pos: int, the position of the baseline condition in the list of DataFrames
    :param compare_pos: int, the position of the comparison condition in the list of DataFrames
    :param compare_condition: str, the name of the comparison condition
    :param estimate: bool, whether to estimate the knowledge data
    :param cutoff: int, the cutoff value for the estimated values
    :return: pd.DataFrame, the preprocessed data

    Assuming the same structure of the behavioral task, here is an example of list of variables:

    behavior_var = ['ReactTime', 'Reward', 'BestOption', 'KeyResponse', 'SetSeen ', 'OptionRwdMean']
    id_var = ['studyResultId', 'Gender', 'Ethnicity', 'Race', 'Age', 'Big5O', 'Big5C', 'Big5E', 'Big5A', 'Big5N',
              'BISScore', 'CESDScore', 'ESIBF_disinhScore', 'ESIBF_aggreScore', 'ESIBF_sScore', 'PSWQScore',
              'STAITScore', 'STAISScore']
    knowledge_var = ['OptionOrder', 'EstOptionA', 'OptionAConfidence', 'EstOptionS', 'OptionSConfidence',
                     'EstOptionK', 'OptionKConfidence', 'EstOptionL', 'OptionLConfidence']
    other_var = id_var + knowledge_var
    numeric_var = ['ReactTime', 'Reward', 'BestOption', 'KeyResponse', 'SetSeen ', 'OptionRwdMean', 'OptionOrder',
                   'EstOptionA', 'OptionAConfidence', 'EstOptionS', 'OptionSConfidence', 'EstOptionK',
                   'OptionKConfidence', 'EstOptionL', 'OptionLConfidence', 'studyResultId', 'Big5O', 'Big5C', 'Big5E',
                   'Big5A', 'Big5N', 'BISScore', 'CESDScore', 'ESIBF_disinhScore', 'ESIBF_aggreScore', 'ESIBF_sScore',
                   'PSWQScore', 'STAITScore', 'STAISScore']
    dict_var = ['Gender', 'Ethnicity', 'Race', 'Age']

    Then, you can call the function like this:

    data_BF, knowledge_BF = preprocess_data('./data/BF', behavior_var, other_var, numeric_var, dict_var,
                                        estimate=True)
    """

    # Define a function to safely parse the dictionary-like string and extract the value
    def extract_value(dict_like_str):
        try:
            # Safely evaluate the string as a dictionary
            val = ast.literal_eval(dict_like_str)
            # Return the value associated with the key 'Q0'
            return val.get('Q0', None)  # Replace 'Q0' with your actual key
        except (ValueError, SyntaxError):
            # In case the string is not a valid dictionary-like string, return NaN or some default value
            return pd.NA

    # This is the function that processes the data for a single participant
    def process_participant_data(participant_path):
        # Initialize an empty list to store the DataFrames
        dfs = []

        # Loop through each folder in the base directory
        for folder_name in os.listdir(participant_path):
            # Construct the full path to the folder
            folder_path = os.path.join(participant_path, folder_name)

            # Check if it is indeed a directory
            if os.path.isdir(folder_path):
                # In each folder, list the .txt files
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.txt'):
                        # Construct the full file path
                        file_path = os.path.join(folder_path, file_name)

                        # Read the JSON file into a DataFrame
                        df = pd.read_json(file_path, lines=True)

                        # Append the DataFrame to the list
                        dfs.append(df)

        dfs[baseline_pos] = dfs[baseline_pos].apply(lambda x: x.explode() if x.name in behavioral_list else x)
        dfs[baseline_pos]['Condition'] = 'Baseline'
        dfs[compare_pos] = dfs[compare_pos].apply(lambda x: x.explode() if x.name in behavioral_list else x)
        dfs[compare_pos]['Condition'] = compare_condition

        # Combine all the DataFrames into one
        combined_df = pd.concat(dfs, ignore_index=True)

        # Select only the columns we need
        kept_columns = behavioral_list + other_data_list + ['Condition']
        combined_df = combined_df[kept_columns]

        # fill in values and drop NA
        combined_df[other_data_list] = combined_df[other_data_list].bfill().ffill()
        combined_df = combined_df.dropna()

        # change the data type of the columns
        combined_df[numeric_list] = (
            combined_df[numeric_list].map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x))

        # Apply the function to the relevant column
        for col in dict_list:
            combined_df[col] = combined_df[col].apply(extract_value)

        return combined_df

    # The below two functions are used only for the knowledge questionnaires
    # Function to reorder and rename option and confidence columns based on 'OptionOrder'
    def reorder_and_rename(row):
        # Reorder options
        order_to_option_col = {order: col for col, order in zip(options_to_pos.keys(), row['OptionOrder'])}
        reordered_options = [row[order_to_option_col[order]] for order in sorted(row['OptionOrder'])]

        # Reorder confidences
        order_to_confidence_col = {order: col for col, order in zip(confidences_to_pos.keys(), row['OptionOrder'])}
        reordered_confidences = [row[order_to_confidence_col[order]] for order in sorted(row['OptionOrder'])]

        return pd.Series(reordered_options + reordered_confidences, index=new_option_cols + new_confidence_cols)

    # Function to clear nonsense data in the knowledge questionnaires
    def confidence_data_preprocessing(data):
        # first, convert all "," to "."
        data = data.str.replace(',', '.')
        # if the value contains a mixture of numbers and letters, extract the numbers
        data = data.str.extract(r'(\d+.\d+|\d+)', expand=False)
        # if the value is still a string, replace it with NaN
        data = pd.to_numeric(data, errors='coerce')
        # then, convert all the values to float
        data = data.astype(float)
        return data

    # List to store the DataFrames for all participants
    all_participants_dfs = []
    i = 0

    # Iterate over each subfolder in the main folder
    for participant_folder_name in os.listdir(main_folder_directory):
        print(f'Processing participant: {i + 1}')
        i += 1

        participant_folder_path = os.path.join(main_folder_directory, participant_folder_name)

        # Check if this path is indeed a folder
        if os.path.isdir(participant_folder_path):
            # Process the participant folder and collect the DataFrame
            participant_df = process_participant_data(participant_folder_path)
            all_participants_dfs.append(participant_df)

    # drop the dfs that are empty
    all_participants_dfs = [df for df in all_participants_dfs if not df.empty]

    # Combine all participant DataFrames into one
    all_data_combined = pd.concat(all_participants_dfs, ignore_index=True)

    # move participant id to the first column
    # Pop the column
    col = all_data_combined.pop('studyResultId')

    # Insert it at the start
    all_data_combined.insert(0, 'studyResultId', col)

    # reset the participant id column
    subject_id = len(all_data_combined) // 400 + 1
    ids = np.arange(1, subject_id)
    ids = np.repeat(ids, 400)
    all_data_combined['studyResultId'] = ids
    all_data_combined = all_data_combined.rename(columns={'studyResultId': 'Subnum'})

    if not estimate:
        return all_data_combined

    if estimate:
        confidence_lists = ['Subnum', 'Condition', 'OptionOrder', 'EstOptionA', 'OptionAConfidence', 'EstOptionS',
                            'OptionSConfidence', 'EstOptionK', 'OptionKConfidence', 'EstOptionL', 'OptionLConfidence']

        confidence_data = (
            all_data_combined[confidence_lists]
            .groupby(['Subnum', 'Condition'])
            .first()
            .reset_index()
        )

        # Mapping of old columns to their positions (1-based for readability)
        options_to_pos = {'EstOptionA': 1, 'EstOptionS': 2, 'EstOptionK': 3, 'EstOptionL': 4}
        confidences_to_pos = {'OptionAConfidence': 1, 'OptionSConfidence': 2, 'OptionKConfidence': 3,
                              'OptionLConfidence': 4}

        # New column names after reordering
        new_option_cols = ['EstA', 'EstB', 'EstC', 'EstD']
        new_confidence_cols = ['A_Confidence', 'B_Confidence', 'C_Confidence', 'D_Confidence']

        # Apply the function to reorder and rename columns for each row
        reordered_values = confidence_data.apply(reorder_and_rename, axis=1)
        reordered_df = pd.concat([confidence_data[['Subnum', 'Condition']].reset_index(drop=True), reordered_values],axis=1)

        # explode the data
        columns_to_explode = new_option_cols + new_confidence_cols
        knowledge_df = reordered_df.explode(columns_to_explode)

        # add a column to indicate phase from 1 to 7
        knowledge_df['Phase'] = knowledge_df.groupby(['Subnum', 'Condition']).cumcount() + 1

        # clear nonsense data
        knowledge_df[new_option_cols] = knowledge_df[new_option_cols].apply(confidence_data_preprocessing)
        knowledge_df[new_confidence_cols] = knowledge_df[new_confidence_cols].apply(confidence_data_preprocessing)

        # if the estimated value is greater than 10, replace it with NaN
        knowledge_df[new_option_cols] = knowledge_df[new_option_cols].apply(lambda x: x.where(x <= cutoff))
        # if the confidence value is not between 1 and 10, replace it with NaN
        knowledge_df[new_confidence_cols] = knowledge_df[new_confidence_cols].apply(
            lambda x: x.where((x >= 1) & (x <= 10)))

        return all_data_combined, knowledge_df


# This function is used to extract numbers from mixed-type columns, such as age
def extract_numbers(data_column):
    result = []
    for item in data_column:
        # If it's a string, extract numeric part using regex
        if isinstance(item, str):
            match = re.search(r'\d+', item)  # Find one or more digits
            if match:
                result.append(float(match.group()))  # Convert to float
            else:
                result.append(np.nan)  # If no match, keep as NaN
        # If it's a number (int or float), append directly
        elif isinstance(item, (int, float)):
            if not np.isnan(item):
                result.append(float(item))  # Convert valid numbers to float
            else:
                result.append(np.nan)  # Preserve NaN
        else:
            result.append(np.nan)  # Handle any other cases as NaN
    return result
