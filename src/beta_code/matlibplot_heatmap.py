import numpy as np
import matplotlib.pyplot as plt

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


model_type = 'T5'
matrix_type = 'pearsonr'

score_matrix_pearsonr,score_matrix_spearmanr = single_model_print(model_split_dict[model_type])

vegetables = col_head
farmers = line_head

if matrix_type == 'pearsonr':
    harvest = np.array(score_matrix_pearsonr)
else:
    harvest = np.array(score_matrix_spearmanr)


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(farmers)), labels=farmers)
ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title(f"model-{model_type}-{matrix_type}-Correlation coefficient")
fig.tight_layout()
plt.show()