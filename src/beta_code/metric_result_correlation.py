import os

from scipy.stats import pearsonr, spearmanr
import pandas as pd
from beta_code.get_human_check_data_eval_score_pairs import mode_ref_dict
from human_metrics import get_score_map
from beta_code.machine_metrics import cal_metrics
import numpy as np
import matplotlib.pyplot as plt

machine_data_base_path = '/eval_data/machine_eval/'
mode_metric_file_dict = {
    2:"rnn_predic.json",
    3:"machine_metric_data_gpt_ep50.json",
    4:"machine_metric_data_unilm.json",
    5:"machine_metric_data_zhouwenwang.json",
    6:"machine_metric_t5_pesg_ep15.json",
    7:"gpt3_base_davinci_metric.json",
    8:"gpt3_davinci_ft_metric.json",
    9:"machine_metric_data_CPM_large.json",
    10:"machine_metric_data_pangu_a.json"
    }


line_head = ['bleu_1','bleu_2','bleu_3','bleu_4','gleu','rouge_1','rouge_2','rouge_l','distinct_1','distinct_2']

# col_head = ['humor_score','fluent_score','comprehensive_score','diss_score']
col_head = ['comprehensive_score','humor_score','fluent_score','diss_score']

col_head_convert = {
    'comprehensive_score':'General','humor_score':'Humor','fluent_score':'Coherence','diss_score':'Ethical-risk'
}

line_head_convert = {
    'bleu_1':'BLEU_1','bleu_2':'BLEU_2','bleu_3':'BLEU_3','bleu_4':'BLEU_4','gleu':'GLEU','rouge_1':'ROUGE_1','rouge_2':'ROUGE_2','rouge_l':'ROUGE_l','distinct_1':'Distinct_1','distinct_2':'Distinct_2'
}

score_map = get_score_map()
def run_metric_correlation(round_size=4):
    all_machine_eval_result_dict = {}
    all_human_eval_result_dict = {}

    score_matrix_pearsonr = [[0 for i in range(len(line_head))] for i in range(len(col_head))]
    score_matrix_spearmanr = [[0 for i in range(len(line_head))] for i in range(len(col_head))]

    for model_idx in range(2,11):
        machine_metrics = cal_metrics(os.path.join(machine_data_base_path, 'data', mode_metric_file_dict[model_idx]))
        human_metrics = score_map.get(mode_ref_dict[model_idx])
        for key in line_head:
            vals = machine_metrics[key]
            if key not in all_machine_eval_result_dict:
                all_machine_eval_result_dict[key] = []
            all_machine_eval_result_dict[key].append(vals)
        for key in col_head:
            vals = human_metrics[key]
            if key not in all_human_eval_result_dict:
                all_human_eval_result_dict[key] = []
            all_human_eval_result_dict[key].append(vals)
    for machine_idx in range(len(line_head)):
        machine_metric_name = line_head[machine_idx]
        for human_idx in range(len(col_head)):
            human_metric_name = col_head[human_idx]
            machine_vector = all_machine_eval_result_dict[machine_metric_name]
            human_vector = all_human_eval_result_dict[human_metric_name]
            P_value = pearsonr(human_vector, machine_vector)[0]
            S_value = spearmanr(human_vector, machine_vector)[0]

            score_matrix_pearsonr[human_idx][machine_idx] = round(P_value,round_size)
            score_matrix_spearmanr[human_idx][machine_idx] = round(S_value,round_size)





    return score_matrix_pearsonr,score_matrix_spearmanr

def out_excel():
    all_export_data = []
    score_matrix_pearsonr,score_matrix_spearmanr = run_metric_correlation()
    fir_line = ['pearsonr'] + line_head
    pearsonr = [fir_line] + [[col_head[i]] + score_matrix_pearsonr[i] for i in range(4)]

    all_export_data.extend(pearsonr)
    all_export_data.append([])

    fir_line = ['spearman'] + line_head
    spearman = [fir_line] + [[col_head[i]] + score_matrix_spearmanr[i] for i in range(4)]

    all_export_data.extend(spearman)
    all_export_data.append([])

    pd.DataFrame(all_export_data).to_excel('metric_correlation.xls')


if __name__ == '__main__':


    # matrix_type = 'pearsonr'
    matrix_type = 'spearman'

    score_matrix_pearsonr,score_matrix_spearmanr = run_metric_correlation(2)

    vegetables = col_head
    farmers = line_head

    if matrix_type == 'pearsonr':
        harvest = np.array(score_matrix_pearsonr)
    else:
        harvest = np.array(score_matrix_spearmanr)

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=[line_head_convert[i] for i in farmers])
    ax.set_yticks(np.arange(len(vegetables)), labels=[col_head_convert[i] for i in vegetables])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, format(harvest[i, j],'.2f'),
                           ha="center", va="center", color="w")

    # ax.set_title(f"{matrix_type}-Correlation coefficient")
    fig.tight_layout()
    # plt.savefig(f"{matrix_type}-Correlation coefficient.svg",dpi=600, format='svg')
    plt.savefig(f"{matrix_type}-Correlation coefficient.pdf",bbox_inches='tight')

