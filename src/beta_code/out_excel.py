import pandas as pd
from beta_code.get_human_check_data_eval_score_pairs import mode_ref_dict
from beta_code.indicator_correlation import load_model_results,load_human_results,correlation

model_r = load_model_results()
human_r = load_human_results()
correlation_result,model_split_dict = correlation(model_r, human_r)

line_head = ['bleu_1','bleu_2','bleu_3','bleu_4','gleu','rouge_1','rouge_2','rouge_l','distinct_1','distinct_2']
col_head = ['h_score','f_score','is_best','d_score']


def single_model_print(model_desc):


    score_matrix_pearsonr = [[0 for i in range(len(line_head))] for i in range(len(col_head))]
    score_matrix_spearmanr = [[0 for i in range(len(line_head))] for i in range(len(col_head))]
    for item in model_desc:
        x_idx = line_head.index(item[1])
        y_idx = col_head.index(item[0])
        score_matrix_pearsonr[y_idx][x_idx] = item[2]
        score_matrix_spearmanr[y_idx][x_idx] = item[3]
    return score_matrix_pearsonr,score_matrix_spearmanr


def run_all_rel_data():

    all_pearson = []
    all_spearmanr = []

    sum_score_matrix_pearsonr,sum_score_matrix_spearmanr = run_combine_rel_data()
    fir_line = ['mean'] + line_head
    pearsonr = [fir_line] + [[col_head[i]] + sum_score_matrix_pearsonr[i] for i in range(4)]
    spearmanr = [fir_line] + [[col_head[i]] + sum_score_matrix_spearmanr[i] for i in range(4)]
    all_pearson.extend(pearsonr)
    all_spearmanr.extend(spearmanr)

    all_pearson.append([])
    all_spearmanr.append([])

    for idx in range(2,11):
        key = mode_ref_dict[idx]
        score_matrix_pearsonr,score_matrix_spearmanr = single_model_print(model_split_dict[key])
        fir_line = [key] + line_head
        pearsonr = [fir_line] + [[col_head[i]] + score_matrix_pearsonr[i] for i in range(4)]
        spearmanr = [fir_line] + [[col_head[i]] + score_matrix_spearmanr[i] for i in range(4)]
        all_pearson.extend(pearsonr)
        all_spearmanr.extend(spearmanr)
        all_pearson.append([])
        all_spearmanr.append([])

    print()
    pd.DataFrame(all_spearmanr).to_excel('spearman.xls')
    pd.DataFrame(all_pearson).to_excel('pearson.xls')

def run_combine_rel_data():
    sum_score_matrix_pearsonr = [[0 for i in range(len(line_head))] for i in range(len(col_head))]
    sum_score_matrix_spearmanr = [[0 for i in range(len(line_head))] for i in range(len(col_head))]
    model_size = 0
    for idx in range(2,11):
        key = mode_ref_dict[idx]
        score_matrix_pearsonr,score_matrix_spearmanr = single_model_print(model_split_dict[key])
        model_size += 1
        for y_idx in range(len(col_head)):
            for x_idx in range(len(line_head)):
                sum_score_matrix_pearsonr[y_idx][x_idx] += score_matrix_pearsonr[y_idx][x_idx]
                sum_score_matrix_spearmanr[y_idx][x_idx] += score_matrix_spearmanr[y_idx][x_idx]

    for y_idx in range(len(col_head)):
        for x_idx in range(len(line_head)):
            sum_score_matrix_pearsonr[y_idx][x_idx] = round(sum_score_matrix_pearsonr[y_idx][x_idx] / model_size,4)
            sum_score_matrix_spearmanr[y_idx][x_idx] = round(sum_score_matrix_spearmanr[y_idx][x_idx] / model_size,4)

    return sum_score_matrix_pearsonr,sum_score_matrix_spearmanr





if __name__ == '__main__':
    run_all_rel_data()
    # run_combine_rel_data()
