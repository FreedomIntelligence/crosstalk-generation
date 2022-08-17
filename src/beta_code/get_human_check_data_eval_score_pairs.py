import os,json,io
import pandas as pd
this_dir = os.path.split(os.path.realpath(__file__))[0]


'''
generate_completions.json       生成的后10段文本
meta_prompt.json                50个前10段文本
score_records.json              评分记录
user_list.json                  评分用户

'''


# 模型id对应表
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
# 打分记录说明
score_desc_dict = {
    'prompt_id':'对应哪一个前10句 meta_prompt.json中的id',
    'model_id':'对应哪一个模型 mode_ref_dict',
    'h_score':'幽默度打分，0~5 高分好',
    'f_score':'是否通顺 通顺1，不通顺0',
    'd_score':'是否侮辱 侮辱1，不侮辱0',
    'is_best':'综合打分 0~5 高分好',
    'user':'评分人'
}

# 生成文本说明
generate_desc_dict = {
    'content':'文本',
    'type':'对应的模型类型 mode_ref_dict',
    'prompt_id':'对应的上文id',



}


if __name__ == '__main__':
    pass