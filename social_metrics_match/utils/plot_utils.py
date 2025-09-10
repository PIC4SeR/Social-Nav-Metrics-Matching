import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
import yaml

home = '/workspaces/hunavsim_devcontainer/src/'
# Load config params for experiments
config = yaml.safe_load(open('params.yaml'))['social_metrics_match']

group_keywords = {
        0: 'Passing',
        1: 'Overtaking',
        2: 'Crossing 1',
        3: 'Crossing 2',
        4: 'Narrow turn',
        5: 'Mixed',
        6: 'Crossing 3',
        7: 'Curious person'
    }

num_groups = 8   # 24 rows divided into groups of 3
num_metrics = 4  # 4 metrics per row
metric_keywords = {
    0: 'Unobtrusiveness',
    1: 'Friendliness',
    2: 'Smoothness',
    3: 'Avoidance Foresight'
}
metric_labels = [metric_keywords[i] for i in range(num_metrics)]
# Label for experiments inside a certain group
experiment_keywords = {
    0: 'First',
    1: 'Second',
    2: 'Third'
}
num_experiments = 3

def plot_avg_std_all_metrics(QM_lab_data, optimal_QM_metrics,weight, mean_survey, std_survey):
    """
    Plots the histogram showing the average values and the standard deviations (shown as error bar)
    8 plots are displayed, one for each scenario
    """
    for group in range(num_groups):
        keyword = group_keywords.get(group, f'Group{group+1}')
        # Assume each of the two arrays has three columns and we want to compare the column means.
        labels = ['First', 'Second', 'Third']
        # Compute mean values for each column for both datasets.
        quant_means = np.mean(QM_lab_data[group*3:(group+1)*3, :], axis=1)
        #qual_means = np.mean(HM_lab_data[group*3:(group+1)*3, :], axis=1)
        ##optimal_quant_means = np.mean(optimal_QM_metrics[group*3:(group+1)*3, :], axis=1)
        optimal_quant_means = np.average(optimal_QM_metrics[group*3:(group+1)*3, :], axis=1,weights=weight)
        survey_means = np.mean(mean_survey[group*3:(group+1)*3, :], axis=1)
        survey_std_means = np.mean(std_survey[group*3:(group+1)*3, :], axis=1)
        
        x = np.arange(len(labels))
        width = 0.2  # Width of each bar
        
        plt.figure(figsize=(8, 5))
        plt.bar(x - width, quant_means, width, label='Normalized Quant Lab Data')
        #plt.bar(x , qual_means, width, label='Normalized Qual Lab Data')
        plt.bar(x, optimal_quant_means, width, label='Optimal Quant Metrics')
        plt.bar(x + width, survey_means, width, label='Mean Survey Data', yerr = survey_std_means)
        plt.xticks(x, labels)
        plt.xlabel('Columns')
        plt.ylabel('Mean Value')
        plt.title(f'Comparison for {keyword}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(home + config["data"]["results_path"] + "/plots/plot_histogram_" + keyword +"_metrics.png")
        plt.show()

def plot_avg_std_all_metrics_subplot(QM_lab_data, optimal_QM_data, weight, mean_survey, std_survey):
        """
        Creates a single figure with a 4x2 grid of bar plots comparing average values and standard deviations
        for each of the 8 groups.
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        colors = ['#77DD77','#4682B4', '#7ED4E0']
        labels = ['Run 1', 'Run 2', 'Run 3']
        width = 0.2

        for group in range(num_groups):
            keyword = group_keywords.get(group, f'Group{group+1}')
            row = group // 4
            col = group % 4
            ax = axes[row, col]
            quant_means = np.mean(QM_lab_data[group*3:(group+1)*3, :], axis=1)
            #qual_means = np.mean(HM_lab_data[group*3:(group+1)*3, :], axis=1)
            optimal_quant_means = np.average(optimal_QM_data[group*3:(group+1)*3, :], axis=1, weights=weight)
            survey_means = np.mean(mean_survey[group*3:(group+1)*3, :], axis=1)
            survey_std_means = np.mean(std_survey[group*3:(group+1)*3, :], axis=1)
            
            x = np.arange(len(labels))
            ax.bar(x - width, quant_means, width, label='QM Lab Data', color=colors[0])
            ax.bar(x, optimal_quant_means, width, label='Optim QM Lab Data', color=colors[1])
            ax.bar(x + width, survey_means, width, label='HM Survey Data', color=colors[2], yerr=survey_std_means, capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=15)
            ax.set_ylabel('Mean Score', fontsize=15)
            ax.set_title(f'{keyword}', fontsize=18)
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=colors[i], label=label) for i, label in enumerate(['QM Lab Data', 'Optimal QM Lab Data', 'HM Survey Data'])]
        fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        plt.savefig(home + config["data"]["results_path"] + "/plots/plot_histogram_subplot_metrics.png")
        plt.show()
def plot_avg_std_survey_metrics(avg_data,std_data):
    """
    Plots the histogram showing the average values and the standard deviations (shown as error bar)
    for the qualitative metrics score given in the survey. 8 plots are displayed, one for each scenario
    The three cases are displayed in the order of the survey if default options for the array extraction functions are kept
    The four groups represent the different qualitative metrics
    First input is the array of average surevy metrics, the second input is the array of the standard deviations
    """
    experiment_labels = [experiment_keywords[i] for i in range(num_experiments)]
    x = np.arange(num_metrics)
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]

    for group in range(num_groups):
        keyword = group_keywords.get(group, f'Group{group+1}')
        group_avg = avg_data[group*3:(group+1)*3, :]
        group_std = std_data[group*3:(group+1)*3, :]

        plt.figure(figsize=(8, 5))
        for i in range(3):
            plt.bar(x + offsets[i],
                    group_avg[i],
                    bar_width,
                    yerr=group_std[i],
                    capsize=5,
                    label=experiment_labels[i])
        
        plt.xticks(x, metric_labels)
        plt.xlabel('Metrics')
        plt.ylabel('Average Value')
        plt.title(f'{keyword} Comparison (Rows {group*3+1} to {group*3+3})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(home + config["data"]["results_path"] + "/plots/plot_survey_group_" + keyword + ".png")
        plt.show()

def plot_tSNE_2D(cluster_data_2d, cluster_labels, name=''):
    """
    Plot clustered data points after t-SNE 2D dim reduction
    """
    plt.figure(figsize=(8,6))
    plt.scatter(cluster_data_2d[:, 0], cluster_data_2d[:, 1],
                c=cluster_labels, cmap='viridis', s=50)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"t-SNE 2D Visualization of k-means Clusters, {name} features")
    plt.colorbar(label="HC Cluster Label")
    plt.savefig(home + config["data"]["results_path"] + "/plots/plot_tSNE_2D_"+name+".png")
    plt.show()
    