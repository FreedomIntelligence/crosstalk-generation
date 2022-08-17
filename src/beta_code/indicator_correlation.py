import json,sys
sys.path.append('.')
from beta_code import machine_metrics
from collections import defaultdict

def get_metrics(references, candidates):
    distinct_1 = machine_metrics.calc_distinct_ngram(candidates, 1)
    distinct_2 = machine_metrics.calc_distinct_ngram(candidates, 2)
    belu_scores = machine_metrics.calculate_bleu_score(references, candidates)
    gleu_scores = machine_metrics.calculate_gleu_score(references, candidates)
    rouge_scores = machine_metrics.calculate_rouge_score(references, candidates)

    result = {
        'bleu_1':belu_scores[0] * 100,
        'bleu_2':belu_scores[1] * 100,
        'bleu_3':belu_scores[2] * 100,
        'bleu_4':belu_scores[3] * 100,
        'gleu':gleu_scores * 100,
        'rouge_1':rouge_scores[0] * 100,
        'rouge_2':rouge_scores[1] * 100,
        'rouge_l':rouge_scores[2] * 100,
        'distinct_1':distinct_1 * 100,
        'distinct_2':distinct_2 * 100
    }

    return result


def load_model_results():
    model_results = json.load(open('../../eval_data/human_eval/data/generate_completions.json'))['RECORDS']
    model_result_group_by_prompt = defaultdict(dict)
    for item in model_results:
        model_result_group_by_prompt[item['prompt_id']][item['type']] = item['content']
    
    model_scores = defaultdict(dict)
    for prompt_id, item in model_result_group_by_prompt.items():
        label = item[1].split('\n')
        for model_id, content in item.items():
            content = content.split('\n')
            model_scores[prompt_id][model_id] = get_metrics(label[:len(content)], content)
    return model_scores
    
def load_human_results():
    human_results = json.load(open('../../eval_data/human_eval/data/score_records.json'))['RECORDS']
    human_result_group_by_prompt = defaultdict(dict)
    for item in human_results:
        if item['model_id'] in human_result_group_by_prompt[item['prompt_id']]:
            human_result_group_by_prompt[item['prompt_id']][item['model_id']]['h_score'].append(item['h_score'])
            human_result_group_by_prompt[item['prompt_id']][item['model_id']]['f_score'].append(item['f_score'])
            human_result_group_by_prompt[item['prompt_id']][item['model_id']]['d_score'].append(item['d_score'])
            human_result_group_by_prompt[item['prompt_id']][item['model_id']]['is_best'].append(item['is_best'])
        else:
            human_result_group_by_prompt[item['prompt_id']][item['model_id']] = {'h_score':[item['h_score']], 'f_score':[item['f_score']], 'd_score':[item['d_score']], 'is_best':[item['is_best']]}

    for prompt_id, result_item in human_result_group_by_prompt.items():
        for model_id, scores in result_item.items():
            human_result_group_by_prompt[prompt_id][model_id]['h_score'] = sum(scores['h_score'])/len(scores['h_score'])
            human_result_group_by_prompt[prompt_id][model_id]['f_score'] = sum(scores['f_score'])/len(scores['f_score'])
            human_result_group_by_prompt[prompt_id][model_id]['d_score'] = sum(scores['d_score'])/len(scores['d_score'])
            human_result_group_by_prompt[prompt_id][model_id]['is_best'] = sum(scores['is_best'])/len(scores['is_best'])

    return human_result_group_by_prompt

def correlation(model_scores, human_scores):
    from human_metrics import mode_ref_dict
    from scipy.stats import pearsonr, spearmanr

    model_prompt_result = defaultdict(lambda:{"human":{"h_score":[], "f_score":[], "d_score":[], "is_best":[]}, "machine":{"bleu_1":[], "bleu_2":[], "bleu_3":[], "bleu_4":[], "gleu":[], "rouge_1":[], "rouge_2":[], "rouge_l":[], "distinct_1":[], "distinct_2":[]}})
    prompt_indexes = list(model_scores.keys())
    result_list = []
    model_split_dict = {}
    for pid in prompt_indexes:
        for mid, model_score in model_scores[pid].items():
            # if mid not in model_prompt_result:
            model_prompt_result[mid]['machine']["bleu_1"].append(model_score["bleu_1"])
            model_prompt_result[mid]['machine']["bleu_2"].append(model_score["bleu_2"])
            model_prompt_result[mid]['machine']["bleu_3"].append(model_score["bleu_3"])
            model_prompt_result[mid]['machine']["bleu_4"].append(model_score["bleu_4"])
            model_prompt_result[mid]['machine']["gleu"].append(model_score["gleu"])
            model_prompt_result[mid]['machine']["rouge_1"].append(model_score["rouge_1"])
            model_prompt_result[mid]['machine']["rouge_2"].append(model_score["rouge_2"])
            model_prompt_result[mid]['machine']["rouge_l"].append(model_score["rouge_l"])
            model_prompt_result[mid]['machine']["distinct_1"].append(model_score["distinct_1"])
            model_prompt_result[mid]['machine']["distinct_2"].append(model_score["distinct_2"])
        # print(model_prompt_result)
        for mid, human_score in human_scores[pid].items():
            # print(human_score)
            model_prompt_result[mid]["human"]["h_score"].append(human_score["h_score"])
            model_prompt_result[mid]["human"]["f_score"].append(human_score["f_score"])
            model_prompt_result[mid]["human"]["d_score"].append(human_score["d_score"])
            model_prompt_result[mid]["human"]["is_best"].append(human_score["is_best"])
    for model_id, model_value in model_prompt_result.items():
        if model_id == 1:
            continue
        for h_score_name, h_score_value in model_value['human'].items():
            for m_score_name, m_score_value in model_value['machine'].items():
                # print(model_id, h_score_name, m_score_name, h_score_value, m_score_value)
                P_value = pearsonr(h_score_value, m_score_value)[0]
                S_value = spearmanr(h_score_value, m_score_value)[0]
                result_list.append([mode_ref_dict.get(model_id), h_score_name, m_score_name, round(P_value, 4), round(S_value, 4)])
                if mode_ref_dict.get(model_id) not in model_split_dict:
                    model_split_dict[mode_ref_dict.get(model_id)] = []
                model_split_dict[mode_ref_dict.get(model_id)].append([h_score_name, m_score_name, round(P_value, 4), round(S_value, 4)])

    return result_list,model_split_dict














print()

# for r_type, r_score in model_value.items():
    #     for score_name, score_value in r_score.items():
    #         print(model_id, r_type, score_name, len(score_value))