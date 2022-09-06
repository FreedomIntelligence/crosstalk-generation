import os,sys,io,json
import pandas as pd

this_dir = os.path.split(os.path.realpath(__file__))[0]

mode_ref_dict = {
    1: "真实数据",
    2:"rnn",
    3:"GPT",
    4:"unilm",
    5:"zhouwenwang",
    6:"T5",
    7:"GPT3",
    8:"GPT3-finetune",
    9:"CPM",
    10:"PANGU-a"
    }

def get_score_map():
    human_eval_scores_file = os.path.join(this_dir,'data','score_records.json')

    score_record = json.loads(io.open(human_eval_scores_file, 'r').read())

    score_map = {}
    for key,records in pd.DataFrame(score_record['RECORDS']).groupby('user'):
        # 只有全部答完的，我们才纳入统计
        if len(records) == 50:
            model_scores = records.groupby('model_id')
            for sub_key,model_score in model_scores:
                h_score = model_score['h_score'].sum()
                f_score = model_score['f_score'].sum()
                d_score = model_score['d_score'].sum()
                is_best = model_score['is_best'].sum()
                model_key = mode_ref_dict[sub_key]
                if model_key in score_map:
                    score_map[model_key]['humor_score'] += h_score
                    score_map[model_key]['fluent_score'] += f_score
                    score_map[model_key]['diss_score'] += d_score
                    score_map[model_key]['comprehensive_score'] += is_best
                else:
                    score_map[model_key] = {
                        'humor_score':h_score,
                        'fluent_score':f_score,
                        'diss_score':d_score,
                        'comprehensive_score':is_best
                    }
    return score_map

if __name__ == '__main__':



    score_map = get_score_map()

    for key in score_map.keys():
        val = score_map.get(key)
        print(key + ":" + str(val))
