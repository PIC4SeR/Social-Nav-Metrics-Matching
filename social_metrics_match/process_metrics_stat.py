
from pathlib import Path
import yaml
import numpy as np
import os
from os.path import expanduser

from utils.data_organization import organize_dict_lab_data, get_all_lab_data_arr, np_extract_exp_lab, np_single_lab_run
from utils.data_organization  import organize_dict_survey, weighted_avg_survey_data, get_robotics_knowledge, datacube_qual_survey_data
from utils.plot_utils import plot_avg_std_survey_metrics, plot_avg_std_all_metrics

home = expanduser("~")
# Load config params for experiments
config = yaml.safe_load(open('params.yaml'))['social_metrics_match']

def main():
    lab_data_path = home + config['data']['repo_dir'] + config['data']['lab_data_path']
    survey_data_path = home + config['data']['repo_dir'] + config['data']['survey_data_path']
    results_dir = home + config['data']['results_path']
    print("lab data path: ", lab_data_path)
    print("survey data path: ", survey_data_path)
    print("results dir path: ", results_dir)

    # Extract LAB data arrays
    dict_lab_data = organize_dict_lab_data(lab_data_path)

    # Extract the np arrays of a specific experiments identified by its keys
    passing_good_QM_array, passing_good_HM_array = np_single_lab_run(dict_lab_data, experiment='Passing', label='Good')
    print(f"Passing single run QM shape:{passing_good_QM_array.shape}, passing single run HM shape: {passing_good_HM_array.shape}")
    print(f"Passing good QM: {passing_good_QM_array},\nPassing good HM: {passing_good_HM_array}") 

    # Extract the np arrays of a lab scenario (all the 3 runs with different labels), dividing QM and HM
    passing_QM_array, passing_HM_array = np_extract_exp_lab(dict_lab_data, experiment='Advanced 4', order=False, normalize=True, normalization="rescale")
    print(f"passing QM shape:{passing_QM_array.shape}, passing HM shape: {passing_HM_array.shape}")
    print(f"passing QM: {passing_QM_array},\npassing HM: {passing_HM_array}")

    # Starting from the complete dataframe with lab data, Extract the np arrays of all lab scenarios dividing QM and HM
    all_lab_QM_array, all_lab_HM_array = get_all_lab_data_arr(dict_lab_data, normalize=True, normalization="rescale")
    print(f"All lab QM array: {all_lab_QM_array.shape}, All lab HM array: {all_lab_HM_array.shape}")
    # print(f"All lab QM array: {all_lab_QM_array}, All lab HM array: {all_lab_HM_array}")

    # Extract SURVEY data arrays
    dict_survey_data = organize_dict_survey(survey_data_path)
    robot_knowledge_array = get_robotics_knowledge(survey_data_path)

    # To extract np arrays of all the survey data
    survey_datacube = datacube_qual_survey_data(dict_survey_data, normalize=True)

    # To directly extract the average and std: If Weighted average set w_avg=True (use robotics background knowledge as weights)
    weighted_survey_array_avg, weighted_survey_array_std = weighted_avg_survey_data(dict_survey_data, robot_knowledge_array, w_avg=True)
    print(f"survey weighted avg shape: {weighted_survey_array_avg.shape},\nsurvey weighted std shape:  {weighted_survey_array_std.shape}") 

    # plot Results
    plot_avg_std_survey_metrics(weighted_survey_array_avg, weighted_survey_array_std)
    plot_avg_std_all_metrics(all_lab_QM_array, all_lab_HM_array, weighted_survey_array_avg, weighted_survey_array_std)


if __name__ == "__main__":
    main()