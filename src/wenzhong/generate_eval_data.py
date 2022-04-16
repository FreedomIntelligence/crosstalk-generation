'''
Author: anon
Date: 2022-02-08 16:12:50
LastEditors: anon
LastEditTime: 2022-02-09 16:07:19
FilePath: /crosstalk-generation/src/gpt/generate_eval_data.py
Description: 

Copyright (c) 2022 by anon/Ultrapower, All Rights Reserved. 
'''
import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config,GPT2Tokenizer
from transformers import BertTokenizerFast
# from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import io
from transformers import pipeline, set_seed
PAD = '[PAD]'
pad_id = 0


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')

    parser.add_argument('--log_path', default='/data1/anon/crosstalk-generation/src/wenzhong/logs/interact.log', type=str, required=False, help='interact日志存放位置')
    
    parser.add_argument('--model_path', default='/data1/anon/crosstalk-generation/pretrain_model/wenzhong', type=str, required=False, help='对话模型路径')
    parser.add_argument('--test_filter_data', default="/data1/anon/crosstalk-generation/src/wenzhong/data/p256-s64/test_filter_50x20.txt", type=str, required=False, help="数据基准文件，n个篇章，每个篇章20行")
    parser.add_argument('--save_samples_path', default="/data1/anon/crosstalk-generation/src/wenzhong/sample/", type=str, required=False, help="保存跑机器指标数据的文件路径及保存篇章级生成的文件路径")
    parser.add_argument('--repetition_penalty', default=2.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=1234, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--utterance_max_len', type=int, default=64, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--seq_max_len', type=int, default=256, help='最大输入长度')
    parser.add_argument('--max_history_len', type=int, default=20, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()




def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate_text_by_input(args,text,history,model,tokenizer):
    
    history.append(text)

    history = [tokenizer.encode(i, add_special_tokens=False) for i in history]

    # input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
    input_ids = []  # 每个input以[CLS]为开头
    history_start_index = 1
    filter_history_sent_ids = []
    for rev_idx in range(len(history)-1,-1,-1):
        
        # this_turn_ids = history[rev_idx][:args.utterance_max_len] + [tokenizer.sep_token_id]
        this_turn_ids = history[rev_idx][:args.utterance_max_len] 
        
        if history_start_index + len(this_turn_ids)  > args.seq_max_len:
            break
        
        filter_history_sent_ids.append(this_turn_ids)
        history_start_index += len(this_turn_ids)
    filter_history_sent_ids.reverse()

    for sent_ids in filter_history_sent_ids:
        input_ids.extend(sent_ids)
        

    input_ids = torch.tensor(input_ids).long().to(model.device)
    input_ids = input_ids.unsqueeze(0)
    response = []  # 根据context，生成的response
    # 最多生成max_len个token
    gen = model.generate(input_ids, max_new_tokens=args.utterance_max_len, do_sample=True, min_length=input_ids.shape[1] + 3,
                        top_p=args.topp,top_k=args.topk,
                        repetition_penalty=args.repetition_penalty,temperature=args.temperature)
    
    text = tokenizer.decode(gen.cpu().numpy().tolist()[0][input_ids.shape[1]:])
    
    return text


def get_machine_metric_datas(model,args,tokenizer):
    '''
    生成机器指标（bleu,gleu,rouge）所需的数据
    A_ori->B_gen
    A_ori,B_ori->C_gen
    A_ori,B_ori,C_ori->D_gen

    最后输出
    B_ori,B_gen
    C_ori,C_gen
    D_ori,D_gen
    '''
    raw_content = io.open(args.test_filter_data,'r').read()
    raw_paras = raw_content.split('\n\n')
    results = []

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    generator.device = model.device
    for single_para in raw_paras:
        single_lines = single_para.split('\n')
        lines_nums = len(single_lines)
        for step in range(1,lines_nums):
            inputs_text = single_lines[:step]
            gen_text = generator('\n'.join(inputs_text),return_full_text=False,num_return_sequences=1,
                                max_new_tokens=64,
                                min_length=3,top_p=args.topp,top_k=args.topk,
                                repetition_penalty=args.repetition_penalty,temperature=args.temperature,pad_token_id=50256
                                )[0]['generated_text']
            # gen_text_tok = generate_text_by_input(args,text, history, model, tokenizer)
            # gen_text = return_seq.replace(inputs_text,'')
            print(gen_text)
            ori_text = single_lines[step]
            results.append({'ori':ori_text,'gen':gen_text})
    
    data_file_path = os.path.join(args.save_samples_path,'machine_metric_data.json')
    io.open(data_file_path,'w').write(json.dumps(results,ensure_ascii=False, indent=4))


    
    
    

def generate_human_check_datas(model,args,tokenizer):
    '''
    生成篇章的方法
    pre_data(:10)->A_gen
    pre_data(:10)+A_gen->B_gen
    pre_data(:10)+A_gen+B_gen->C_gen

    最后输出
    pre_data(:10)
    A_gen
    B_gen
    C_gen
    ...

    
    '''
    pass


def interact(args,samples_file,model,tokenizer):
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        try:
            text = input("user:")
            # text = "你好"
            if args.save_samples_path:
                samples_file.write("user:{}\n".format(text))
            text = generate_text_by_input(args,text,history,model,tokenizer)
            print("chatbot:" + "".join(text))
            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break

def main():
    args = set_args()
    set_random_seed(args)
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储
    get_machine_metric_datas(model,args,tokenizer)
    


if __name__ == '__main__':
    main()
