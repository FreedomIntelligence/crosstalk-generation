'''
Author: anon
Date: 2022-01-27 11:39:10
LastEditors: anon
LastEditTime: 2022-02-08 17:28:13
FilePath: /crosstalk-generation/src/gpt/metrics.py
Description: 

Copyright (c) 2022 by anon/Ultrapower, All Rights Reserved. 
'''
from statistics import mean
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge 
import io,json
# import nltk
# nltk.download('wordnet')
'''
description: 计算bleu1,2,3,4的值
param {原始句} reference
param {预测句} hypothesis
return bleu1,bleu2,bleu3,bleu4
'''
def calculate_bleu_score(references, candidates):

    smooth = SmoothingFunction()

    reference = [[[j for j in i]] for i in references]
    hypothesis = [[j for j in i] for i in candidates]

    BLEU_1 = corpus_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
    BLEU_2 = corpus_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
    BLEU_3 = corpus_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth.method1)
    BLEU_4 = corpus_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    return BLEU_1,BLEU_2,BLEU_3,BLEU_4

'''
description: 计算gleu值
param {原始句} reference
param {预测句} hypothesis
return gleu值
'''
def calculate_gleu_score(references, candidates):


    reference = [[[j for j in i]] for i in references]
    hypothesis = [[j for j in i] for i in candidates]
    return corpus_gleu(reference, hypothesis)

'''
description: 中文不建议使用，因为依赖了wordnet，wordnet是英文词典
param {原始句} reference
param {预测句} hypothesis
return metetor值
'''
def calculate_meteor_score(references, candidates):
    reference = [[[j for j in i]] for i in references]
    hypothesis = [[j for j in i] for i in candidates]
    all_meteor = []
    for ref,hyp in zip(reference, hypothesis):
        all_meteor.append(meteor_score(ref,hyp))
    return mean(all_meteor)


'''
description: rouge值计算
param {原始句} reference
param {预测句} hypothesis
return rouge1，rouge2,rougel
'''
def calculate_rouge_score(reference, hypothesis):

    rouge = Rouge()
    scores = []
    for ref,hyp in zip(reference,hypothesis):
         scores.append(rouge.get_scores(' '.join([i for i in hyp]), ' '.join([i for i in ref])))
    rouge_1 = [i[0]['rouge-1']['f'] for i in scores]
    rouge_2 = [i[0]['rouge-2']['f'] for i in scores]
    rouge_l = [i[0]['rouge-l']['f'] for i in scores]
    return mean(rouge_1),mean(rouge_2),mean(rouge_l)


def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    pair_list = [[j for j in i] for i in pair_list]

    def get_dict(tokens, ngram, gdict=None):
        """
        get_dict
        统计n-gram频率并用dict存储
        """
        token_dict = {}
        if gdict is not None:
            token_dict = gdict
        tlen = len(tokens)
        for i in range(0, tlen - ngram + 1):
            ngram_token = "".join(tokens[i:(i + ngram)])
            if token_dict.get(ngram_token) is not None: 
                token_dict[ngram_token] += 1
            else:
                token_dict[ngram_token] = 1
        return token_dict

    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1 
        #if freq == 1:
        #    ngram_distinct_count += freq
    return ngram_distinct_count / ngram_total



def test_demo():
    references = ['我今天晚上必须回家吃饭','广东鸡翅膀，我最爱吃','天天都需要你爱']
    candidates = ['晚上我要回家吃饭','最好吃的是广东鸡翅膀','啦啦啦啦要你爱']

    
    belu_scores = calculate_bleu_score(references,candidates)
    gleu_scores = calculate_gleu_score(references,candidates)
    # meteor_scores = calculate_meteor_score(references,candidates)
    rouge_scores = calculate_rouge_score(references,candidates)
    print(belu_scores)


def cal_metrics():
    machine_gen_file = '/data1/anon/crosstalk-generation/src/gpt/sample/machine_metric_data.json'
    raw_text = io.open(machine_gen_file,'r').read()
    data_list = json.loads(raw_text)
    references = []
    candidates = []
    for data_item in data_list:
        references.append(data_item['ori'])
        candidates.append(data_item['gen'])
    

    distinct_1 = calc_distinct_ngram(candidates,1)
    distinct_2 = calc_distinct_ngram(candidates,2)
    belu_scores = calculate_bleu_score(references,candidates)
    gleu_scores = calculate_gleu_score(references,candidates)
    rouge_scores = calculate_rouge_score(references,candidates)

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


if __name__ == '__main__':
    

    print(cal_metrics())

    
