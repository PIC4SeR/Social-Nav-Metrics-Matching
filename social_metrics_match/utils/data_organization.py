# This file contains the functions to extract the data from the file_path file and organize it in a pandas dataframe
import pandas as pd
import os
import re
import numpy as np

def extract_data_to_df(file_path):
    """
    Get the initial data out of a tabular file and return it as a pandas dataframe.
    Input : ods or xlsx file
    Output: pd.dataframe
    """
    if file_path.lower().endswith('.ods'):
    # For ODS files
        df_lab_data = pd.read_excel(file_path, engine="odf")
    elif file_path.lower().endswith('.xlsx'):
    # For Excel files (.xlsx)
        df_lab_data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use either ODS or XLSX.")
    return df_lab_data

################################
###### Lab Data Methods ########
################################

def get_init_dict_lab_data(lab_df : pd.DataFrame):
    """
    Handle the dataframe to have a dictionary with the data organized by the experiment name
    Input : pd.dataframe
    Output: dict
    """

    mapping = {
    "Unnamed: 2": "labels",
    "Unnamed: 3": "algorithms",
    "Unnamed: 4": "time to goal",
    "Unnamed: 5": "path length",
    "Unnamed: 6": "cumulative heading changes",
    "Unnamed: 7": "average robot linear speed",
    "Unnamed: 8": "social work",
    "Unnamed: 9": "social work per second",
    "Unnamed: 10": "average minimum distance",
    "Unnamed: 11": "intimate space intrusion",
    "Unnamed: 12": "personal space intrusion",
    "Unnamed: 13": "social space intrusion",
    "Unnamed: 14": "public space occupancy",
    "Unnamed: 15": "qualitative metrics",
    "Unnamed: 16": "unobtrusiveness",
    "Unnamed: 17": "friendliness",
    "Unnamed: 18": "smoothness",
    "Unnamed: 19": "avoidance foresight"
    }

    groups = {}
    current_key = None
    # Iterate over every row in the DataFrame
    for _, row in lab_df.iterrows():
        row_data = row.iloc[2:].to_dict()
        # Remap keys using the mapping dictionary.
        row_dict = {mapping.get(k, k): v for k, v in row_data.items()}
        if pd.notnull(row["Unnamed: 1"]):
            current_key = row["Unnamed: 1"]
            groups[current_key] = [row_dict]
        else:
            if current_key is not None:
                groups[current_key].append(row_dict)
    df_dict_lab_data = groups
    return df_dict_lab_data


def separate_by_label_lab_data(initial_dict : dict):
    """
    This separates the data by the label ["Good", "Mid", "Bad"] creating another nested dictionary
    """
    result = {}
    for experiment, records in initial_dict.items():
        result[experiment] = {"Good": {}, "Mid": {}, "Bad": {}}
        for record in records:
            label = record.get("labels")
            if label in result[experiment]:
                # Using the experiment name and label as separate subgroups
                # If the group key under the label doesn't exist, create it.
                group_key = experiment  # Can modify this if grouping within experiment is needed
                if group_key not in result[experiment][label]:
                    result[experiment][label] = []
                result[experiment][label].append(record)
    return result


def separate_HM_QM_dict_lab_data(data : dict):
    """
    This function separates the data into quantitative and qualitative data, ending up with a dictionary for quantitative and qualitative data
    Every entry is its own dict

    For each experiment and label grouping, split each record into two subdictionaries:
    'quantitative' (all keys from quantitative_keys) and 'qualitative' (all keys from qualitative_keys).
    The resulting structure maps each record to a dictionary holding both subdictionaries.
    Adjust qualitative_keys and quantitative_keys as needed.
    """
    qualitative_keys = {"unobtrusiveness", "friendliness", "smoothness", "avoidance foresight"}
    quantitative_keys = {"time to goal", "path length", "cumulative heading changes", "average robot linear speed",
                          "social work", "social work per second", "average minimum distance",
                          "intimate space intrusion","personal space intrusion","social space intrusion","public space occupancy"
                         }
    new_data = {}
    for experiment, label_groups in data.items():
        new_data[experiment] = {}
        for label, records in label_groups.items():
            split_record = {"quantitative": {}, "qualitative": {}}
            for record in records:
                quantitative_dict = {}
                qualitative_dict = {}
                for key, value in record.items():
                    if key in qualitative_keys:
                        qualitative_dict[key] = value
                    elif key in quantitative_keys:
                        quantitative_dict[key] = value
                split_record["quantitative"] = quantitative_dict
                split_record["qualitative"] = qualitative_dict
            new_data[experiment][label] = split_record
    return new_data

def rescale(arr : np.ndarray):
    """
    Rescale array linearly, such that best val --> 1
    """
    new_array = arr.copy() # Shape: [11, 3] --> [n_metrics, n_run]
    
    for i in range(new_array.shape[0]):
        if i in [7, 8, 9, 10]: # proxemics need to be rescaled
            new_array[i] = 100*np.ones_like(new_array[i]) - new_array[i]

        # metrics that follow best value is the highest: normalized score = (each value / max)
        # average minimum distance to the closest person and proxemics
        if i in list(range(6, 9)):
            best_value = new_array[i].max()
            if best_value < 1e-6:
                new_array[i] = np.zeros_like(new_array[i])
            else:
                new_array[i] = new_array[i] / best_value
        
        else: # metrics that follow best value is the lowest: normalized score = (min / each value)
            best_value = new_array[i].min()
            if new_array[i].any() < 1e-6:
                new_array[i] = np.ones_like(new_array[i])
            else:
                new_array[i] = (best_value / new_array[i])
        

    return new_array

def min_max_normalize(arr: np.ndarray):
    """
    Normalize array with min-max strategy
    """
    new_array = arr.copy() # Shape: [11, 3] --> [n_metrics, n_run]
    
    for i in range(new_array.shape[0]):
        if np.abs(np.max(arr[i]) - np.min(arr[i])) < 1e-6:
            new_array[i] = np.ones_like(new_array[i])
            continue

        norm_arr = (arr[i] - np.min(arr[i])) / (np.max(arr[i]) - np.min(arr[i]))
        if i in list(range(6, 9)): # metrics that follow: best value is the highest
            new_array[i] = norm_arr
        else:  # metrics that follow: best value is the lowest
            new_array[i] = np.ones_like(norm_arr) - norm_arr
        
    return new_array

def normalize_quant_data(quant_metrics_arr : np.ndarray, normalization : str ="rescale"):
    """ 
    Normalize the quantitative metrics with linear rescale in [1, min] or min-max normalization in [0,1]
    """
    # Shape: [11, 3] --> [n_metrics, n_run]
    # print("Normalizing QM data with normalization: ", normalization)
    if normalization != "rescale" and normalization != "min-max":
        raise Exception("normalization should be rescale or min-max")
    if normalization == "rescale":
        new_array = rescale(quant_metrics_arr)
    elif normalization == "min-max":
        new_array = min_max_normalize(quant_metrics_arr)
    return new_array

def organize_dict_lab_data(data):
    """
    Exectutes all the actions to get organized np arrays of lab data:
    1. Getting the data structure from lab data 
    2. Separate dictionary according to labels and HM/QM
    3. Get the np arrays 
    """
    df_lab_data = extract_data_to_df(data)
    df_dict_lab_data = get_init_dict_lab_data(df_lab_data)
    separated_data = separate_by_label_lab_data(df_dict_lab_data)
    organized_data = separate_HM_QM_dict_lab_data(separated_data)
    return organized_data

def np_single_lab_run(lab_dict : dict, experiment : str, label: str, normalizeHM: bool = True):
    """
    Extract np arrays of a single lab experiment from the overall organized dictionary
    """

    # Extract the data for the specified experiment and label
    experiment_data = lab_dict[experiment]
    if experiment_data is None:
        raise ValueError("Experiment not found.")
    label_data = experiment_data[label]
    if label_data is None:
        raise ValueError("Label not found in the specified experiment.")
    quantitative_dict = label_data["quantitative"]
    qualitative_dict = label_data["qualitative"]

    quantitative_data = np.array([v for v in quantitative_dict.values()])
    if normalizeHM:
        qualitative_data = np.array([v for v in qualitative_dict.values()])/5
    else:
        qualitative_data = np.array([v for v in qualitative_dict.values()])

    return quantitative_data, qualitative_data

def np_extract_exp_lab(lab_dict : dict, experiment : str, order : bool = False, normalizeQM : bool = True, normalizeHM : bool = True, normalization : str = "rescale"):
    """
    this function will return the data as two numpy arrays:
    one for quantitative data and the other for qualitative data, the order is given the order the quantitative and qualitative keys above are defined
    by default the columns are put in the order Good, Mid, Bad, but this can be changed by setting the order parameter to True, which assigns every experiment a different order
    based on the order proposed in the survey
    """
    if order == True:
        match experiment:
            case "Passing":
                labels = ["Bad", "Mid", "Good"]
            case "Overtaking":
                labels = ["Good", "Mid", "Bad"]
            case "Crossing 1":
                labels = ["Mid", "Bad", "Good"]
            case "Crossing 2":
                labels = ["Mid", "Good", "Bad"]
            case "Advanced 1":
                labels = ["Bad", "Mid", "Good"]
            case "Advanced 2":
                labels = ["Good", "Mid", "Bad"]
            case "Advanced 3":
                labels = ["Mid", "Bad", "Good"]
            case "Advanced 4":
                labels = ["Mid", "Good", "Bad"]
    else:
        labels = ["Good", "Mid", "Bad"]
    quant_arr = np.array([])
    qual_arr = np.array([])
    for label in labels:
        quant_arr_single, qual_arr_single = np_single_lab_run(lab_dict, experiment, label, normalizeHM=normalizeHM)
        if quant_arr.size == 0:
            quant_arr = quant_arr_single.reshape(-1, 1)
        else:
            quant_arr = np.column_stack((quant_arr, quant_arr_single.reshape(-1, 1)))
        if qual_arr.size == 0:
            qual_arr = qual_arr_single.reshape(-1, 1)
        else:
            qual_arr = np.column_stack((qual_arr, qual_arr_single.reshape(-1, 1)))
    
    # Normalize quantitative data
    if normalizeQM:
        quant_arr = normalize_quant_data(quant_arr, normalization=normalization)
    
    # There is a missing metrics in "Advanced 4" scenario
    if experiment == "Advanced 4":
        qual_arr[0] = np.zeros_like(qual_arr[0])

    return quant_arr, qual_arr

def get_all_lab_data_arr(complete_lab_dict : dict, order : bool = True, normalizeQM : bool = True, normalizeHM : bool = True, normalization : str ="rescale"):
    """ 
    Normalize the quantitative metrics with linear rescale in [1, min] or min-max normalization in [0,1]
    """
    overall_quantitative_array = np.array([])
    overall_qualitative_array = np.array([])
    for key in complete_lab_dict: # loop over all exp scenarios
        # Extract arrays for each exp scenario
        quant_arr, qual_arr = np_extract_exp_lab(complete_lab_dict, key, order=True, normalizeQM=normalizeQM, normalizeHM=normalizeHM, normalization=normalization)

        # stack quantitative data array for each experiment
        if overall_quantitative_array.size == 0:
            overall_quantitative_array = quant_arr
        else:
            overall_quantitative_array = np.hstack((overall_quantitative_array, quant_arr.reshape(-1,3)))

        # stack quantitative data array for each experiment
        if overall_qualitative_array.size == 0:
            overall_qualitative_array = qual_arr
        else:
            expected_rows = overall_qualitative_array.shape[0]
            if qual_arr.shape[0] < expected_rows: # check if number of metrics feature is the expected one (last exp has on less)
                pad_rows = np.full((expected_rows - qual_arr.shape[0], qual_arr.shape[1]), 0.)
                qual_arr = np.vstack((pad_rows, qual_arr))
            overall_qualitative_array = np.hstack((overall_qualitative_array,qual_arr.reshape(-1,3)))

    return overall_quantitative_array.T, overall_qualitative_array.T

################################
###### Survey Data Methods #####
################################

def get_initial_survey_dict(survey_df : pd.DataFrame):
    """
    Transform the data frame with survey results in a dictionary with keys corresponding to each scenario
    For each scenario we have N-rows = n. answers, and M-columns = n. metrics * n. exp runs
    """
    survey_df = survey_df.rename(columns=lambda col: col.replace("A. foresight", "Avoidance foresight"))
    base = "Could you rate your background knowledge in Robotics and Autonomous Navigation"
    repetitions = {}
    for col in survey_df.columns:
        if col in ["Informazioni cronologiche", base, "Colonna 95"]:
            continue
        lc = col.lower()
        # If the column name indicates a passing question
        if "passing" in lc:
            rep = "passing"
            col_base = col.replace("Passing", "").replace("passing", "").strip()
        # If the column name indicates an overtaking question
        elif "overtaking" in lc:
            rep = "overtaking"
            col_base = col.replace("Overtaking", "").replace("overtaking", "").strip()
        else:
            # Use the standard dot-suffix method.
            if "." in col:
                rep = col.rsplit(".", 1)[-1]
                col_base = col.rsplit(".", 1)[0]
            else:
                rep = "0"
                col_base = col
        repetitions.setdefault(rep, {})[col_base] = survey_df[col]

    # Build a list of dataframes: one for each repetition group
    dfs_survey_data = []
    for rep, data in repetitions.items():
        # Create a temporary dataframe for this repetition group
        temp = pd.DataFrame(data)
        # Add the base column (background knowledge) and (optionally) the original timestamp
        temp[base] = survey_df[base]
        temp["Informazioni cronologiche"] = survey_df["Informazioni cronologiche"]
        # Tag with the repetition number
        temp["repetition"] = rep
        dfs_survey_data.append(temp)
    experiment_names = ["Passing", "Overtaking", "Crossing 1", "Crossing 2", "Advanced 1", "Advanced 2", "Advanced 3", "Advanced 4"]
    dfs_dict_survey_data = {}
    for i, exp in enumerate(dfs_survey_data):
        dfs_dict_survey_data[experiment_names[i]] = exp
    
    return dfs_dict_survey_data

def split_survey_data_by_case(dfs : pd.DataFrame):
    """
    For each experiment in the survey data dictionary, split its dataframe by remapping columns whose names start
    with "First", "Second" or "Third" (followed by "case").
    The remaining part of the column name is used as the new column name.
    """
    split_data = {}
    pattern = r"^(First|Second|Third)\s+case\s+(.*)$"
    for exp, df in dfs.items():
        # Prepare empty DataFrames with the same index as the original one.
        sub_dict = {"First": pd.DataFrame(index=df.index),
                    "Second": pd.DataFrame(index=df.index),
                    "Third": pd.DataFrame(index=df.index)}
        for col in df.columns:
            match = re.match(pattern, col)
            if match:
                case_part, metric_name = match.groups()
                sub_dict[case_part][metric_name] = df[col]
        split_data[exp] = sub_dict
    return split_data

def np_single_survey_run(survey_dict : dict, experiment : str, label : str):
    """Extract the data for the specified experiment and label"""
    #the survey data are not technically labeled, in the first passage the labels to be used are "First","Second","Third"
    experiment_data = survey_dict[experiment]
    if experiment_data is None:
        raise ValueError("Experiment not found.")
    label_data = experiment_data[label]
    if label_data is None:
        raise ValueError("Label not found in the specified experiment.")
    # if using the list version of separate_quantitative_qualitative, use this
    #quantitative_data = np.array(list(quantitative_dict))
    #qualitative_data = np.array(list(qualitative_dict))
    qualitative_data = label_data.to_numpy()

    return qualitative_data

def np_extract_exp_survey(survey_dict : dict, experiment : str, order : bool =True):
    """
    Extract np arrays for a specific scenario, getting all the three runs.
    """
    if not order:
        match experiment:
            case "Passing":
                labels = ["Third", "Second", "First"]
            case "Overtaking":
                labels = ["First", "Second", "Third"]
            case "Crossing 1":
                labels = ["Third", "First", "Second"]
            case "Crossing 2":
                labels = ["Second", "First", "Third"]
            case "Advanced 1":
                labels = ["Third", "Second", "First"]
            case "Advanced 2":
                labels = ["First", "Second", "Third"]
            case "Advanced 3":
                labels = ["Third", "First", "Second"]
            case "Advanced 4":
                labels = ["Second", "First", "Third"]
    else:
        labels = ["First", "Second", "Third"]
    qual_arr = np.array([])
    if experiment == "Advanced 4":
        dim = 3
    else:
        dim = 4
        
    for label in labels:
        qual_arr_single = np_single_survey_run(survey_dict, experiment, label)
        if qual_arr.size == 0:
            qual_arr = qual_arr_single.reshape(-1, dim)
        else:
            qual_arr = np.column_stack((qual_arr, qual_arr_single.reshape(-1, dim)))
    return qual_arr


def overall_array_survey_data_all_agents(survey_dict : dict, order=True):
    overall_qual_arr = np.array([])
    for key in survey_dict:
        qual_arr = np_extract_exp_survey(survey_dict,key,order=order)

        if overall_qual_arr.size == 0:
            overall_qual_arr = qual_arr
        else:
            expected_rows = qual_arr.shape[0]
            if qual_arr.shape[0] < expected_rows:
                pad_rows = np.full((expected_rows - qual_arr.shape[0], qual_arr.shape[1]), np.nan)
                qual_arr = np.vstack((pad_rows, qual_arr))
            overall_qual_arr = np.hstack((overall_qual_arr,qual_arr))
    return overall_qual_arr


def datacube_qual_survey_data(survey_dict : dict, normalize: bool = True):
    """
    Extract np arrays with all survey data.
    Input : dictionary with all survey data
    Output: np arrays with shape [n_answers, n_exp_runs, n_metrics] --> [69, 24, 4]
    """
    big_array = overall_array_survey_data_all_agents(survey_dict) #Shape [69, 93]
    column_names = []
    for experiment in survey_dict:
        for case in survey_dict[experiment]:
            if experiment != 'Advanced 4':
                column_names.append(f"{experiment}_{case}_unobtrusiveness")
            column_names.append(f"{experiment}_{case}_friendliness")
            column_names.append(f"{experiment}_{case}_smoothness")
            column_names.append(f"{experiment}_{case}_avoidance foresight")
    df_experiment = pd.DataFrame(big_array, columns=column_names) 
    n_agents = df_experiment.shape[0]
    datacube = np.full((n_agents, 24, 4), np.nan)

    experiments_order = ['Passing', 'Overtaking', 'Crossing 1', 'Crossing 2',
                         'Advanced 1', 'Advanced 2', 'Advanced 3', 'Advanced 4']
    case_order = ['First', 'Second', 'Third']

    for exp_idx, exp in enumerate(experiments_order):
        for case_idx, case in enumerate(case_order):
            cube_case = exp_idx * 3 + case_idx
            if exp == 'Advanced 4':
                # "Advanced 4" has no unobtrusiveness metric, so keep nan for the first metric
                col_friend = f"{exp}_{case}_friendliness"
                col_smooth = f"{exp}_{case}_smoothness"
                col_avoid = f"{exp}_{case}_avoidance foresight"
                if col_friend in df_experiment.columns:
                    datacube[:, cube_case, 0] = np.zeros_like(datacube[:, cube_case, 0])
                    datacube[:, cube_case, 1] = df_experiment[col_friend]
                    datacube[:, cube_case, 2] = df_experiment[col_smooth]
                    datacube[:, cube_case, 3] = df_experiment[col_avoid]
            else:
                col_unob = f"{exp}_{case}_unobtrusiveness"
                col_friend = f"{exp}_{case}_friendliness"
                col_smooth = f"{exp}_{case}_smoothness"
                col_avoid = f"{exp}_{case}_avoidance foresight"
                if col_unob in df_experiment.columns:
                    datacube[:, cube_case, 0] = df_experiment[col_unob]
                    datacube[:, cube_case, 1] = df_experiment[col_friend]
                    datacube[:, cube_case, 2] = df_experiment[col_smooth]
                    datacube[:, cube_case, 3] = df_experiment[col_avoid]
    
    if normalize:
        return datacube / 5
    else:
        return datacube


def weighted_avg_survey_data(survey_dict : dict, robotics_know : np.ndarray, w_avg : bool =True, normalize : bool =True):
    """
    starting from the data dicitionary of survey data and column with people robotics knowledge, compute the standard average or weighted average
    Output: mean array, std array
    """
    datacube = datacube_qual_survey_data(survey_dict, normalize=normalize)

    ## Compute weights from robo_knowledge_survey_data
    if w_avg == True:
        weights = robotics_know  # shape: (n_agents,)
    else:
        weights = np.ones_like(robotics_know, dtype=float)
    
    # Multiply each agent's data in datacube by its corresponding weight.
    # datacube has shape (n_agents, 24, 4); we'll compute a weighted average across agents.
    weighted_sum = np.nansum(datacube * weights[:, None, None], axis=0)
    
    # For each entry, sum of weights ignoring NaNs.
    weights_sum = np.nansum(weights[:, None, None] * (~np.isnan(datacube)), axis=0)
    
    # Compute weighted average for each (case, metric)
    avg_per_case = weighted_sum / weights_sum  # shape: (24, 4)
    
    # Calculate weighted standard deviation:
    diff = datacube - avg_per_case[None, :, :]
    weighted_var = np.nansum(weights[:, None, None] * diff**2, axis=0) / weights_sum
    std_per_case = np.sqrt(weighted_var)  # shape: (24, 4)
    
    return avg_per_case, std_per_case

def organize_dict_survey(data_path):
    """
    Exectutes all the actions to get organized np arrays of survey data:
     1. Get the data structure from survey data 
     2. Split survey data by case
     3. Get the np arrays
    """
    df_survey_data = extract_data_to_df(data_path)
    df_dict_survey_data = get_initial_survey_dict(df_survey_data)
    separated_survey_data = split_survey_data_by_case(df_dict_survey_data)
    return separated_survey_data

def get_robotics_knowledge(data_path):
    """
    Extract column with robotics background knowledge scores
    """
    df_survey_data = extract_data_to_df(data_path)
    df_dict_survey_data = get_initial_survey_dict(df_survey_data)
    col_name = "Could you rate your background knowledge in Robotics and Autonomous Navigation"
    column = df_dict_survey_data["Passing"][col_name].to_numpy()
    return column