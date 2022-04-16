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
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
# from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import io,sys

sys.path.append('/data1/anon/crosstalk-generation/src/zhouwenwang/Fengshenbang-LM')



from fengshen import RoFormerModel    
from fengshen import RoFormerConfig
from transformers import BertTokenizer


PAD = '[PAD]'
pad_id = 0


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1.2, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=2, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0.3, type=float, required=False, help='最高积累概率')

    parser.add_argument('--log_path', default='/data1/anon/crosstalk-generation/src/zhouwenwang/logs/interact.log', type=str, required=False, help='interact日志存放位置')
    
    parser.add_argument('--model_path', default='/data1/anon/crosstalk-generation/pretrain_model/zhouwenwang', type=str, required=False, help='对话模型路径')
    parser.add_argument('--test_filter_data', default="/data1/anon/crosstalk-generation/src/zhouwenwang/data/p256-s64/test_filter_50x20.txt", type=str, required=False, help="数据基准文件，n个篇章，每个篇章20行")
    parser.add_argument('--save_samples_path', default="/data1/anon/crosstalk-generation/src/zhouwenwang/sample/", type=str, required=False, help="保存跑机器指标数据的文件路径及保存篇章级生成的文件路径")
    parser.add_argument('--repetition_penalty', default=2.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--diverse_penalty', default=2.0, type=float, required=False,
                        help="历史出现字惩罚项")
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

    input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
    history_start_index = 1
    filter_history_sent_ids = []
    for rev_idx in range(len(history)-1,-1,-1):
        
        this_turn_ids = history[rev_idx][:args.utterance_max_len] + [tokenizer.sep_token_id]
        # this_turn_ids = history[rev_idx][:args.utterance_max_len] 
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
    for idx in range(args.utterance_max_len):
        outputs = model(input_ids=input_ids)
        logits = outputs[0]
        logits = torch.nn.functional.linear(
                logits, model.embeddings.word_embeddings.weight)
        next_token_logits = logits[0, -1, :]
        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
        for id in set(response):
            next_token_logits[id] /= args.repetition_penalty
        for id in set(input_ids.cpu().numpy()[0].tolist()):
            next_token_logits[id] /= args.diverse_penalty
        next_token_logits = next_token_logits / args.temperature
        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
        # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        no_need_tok = [tokenizer.cls_token_id,
                            
                            tokenizer.sep_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.mask_token_id,
                            tokenizer.convert_tokens_to_ids('#'),
                            tokenizer.convert_tokens_to_ids('。'),
                            tokenizer.convert_tokens_to_ids('?'),
                            
                            tokenizer.convert_tokens_to_ids('.')
                            
                            ] + [i for i in range(192)]
        
        
        if (next_token.cpu().numpy()[0] in no_need_tok) :  # 遇到[SEP]则表明response生成结束
            if len(response) > 0:
                break
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=len(no_need_tok) + 2)[1:]
                for candi_tok in next_token:
                    if not candi_tok in no_need_tok:
                        next_token = candi_tok.reshape(1,)
                        break
        
        if next_token == 0:
            continue
        response.append(next_token.item())
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
        # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
        # print("his_text:{}".format(his_text))
    if len(response) == 0:
        print('')
    
    text = tokenizer.convert_ids_to_tokens(response)
    text_str = "".join(text)
    return text



def single_line(args,sentence,model,tokenizer):
    new_sent = ''
    unused_ids = [i for i in range(177)] + [i for i in range(181,192)] + [i for i in range(7681,12000)]
    history_ids = []
    for i in range(args.utterance_max_len):
        encode = torch.tensor(
            [[tokenizer.cls_token_id]+tokenizer.encode(sentence, add_special_tokens=False)]).cuda()
        
        logits = model(encode)[0]
        logits = torch.nn.functional.linear(
            logits, model.embeddings.word_embeddings.weight)

        for res_id in history_ids:
            logits[0][-1][res_id] = logits[0][-1][res_id] / args.repetition_penalty
        for unuse_id in unused_ids:
            logits[0][-1][unuse_id] = -float('Inf')
        logits = torch.nn.functional.softmax(
            logits, dim=-1).cpu().detach().numpy()[0]

        prob = logits[-1]
        

        tok_id = int(np.random.choice(logits.shape[1], p=prob))
        history_ids.append(tok_id)
        gen_txt = tokenizer.decode(tok_id)
        sentence = sentence + gen_txt
            
        new_sent = new_sent + gen_txt
        if len(new_sent) > 1 and new_sent[-1] in ['。','！','？','…']:
            break
    return new_sent

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
    
    for single_para in raw_paras:
        single_lines = single_para.split('\n')
        lines_nums = len(single_lines)
        for step in range(1,lines_nums):
            inputs_text = single_lines[:step]
            history = inputs_text[:-1]
            text = inputs_text[-1]
            gen_text_tok = generate_text_by_input(args,text, history, model, tokenizer)
            gen_text = "".join(gen_text_tok)
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
    raw_content = io.open(args.test_filter_data,'r').read()
    raw_paras = raw_content.split('\n\n')
    results = []

    # text_generator = TextGenerationPipeline(model, tokenizer,device=0)
    
    for single_para in raw_paras:
        single_lines = single_para.split('\n')
        lines_nums = len(single_lines)
        history_all = single_lines[:10]
        for step in range(10,lines_nums):
            
            inputs_text = single_lines[:step]
            
            gen_text_tok = single_line(args, history_all[-1] , model, tokenizer)

            # inputs_text = history_all[:step]
            # gen_text_tok = single_line(args, ''.join(inputs_text) , model, tokenizer)
            # gen_text = "".join(gen_text_tok)
            gen_text = gen_text_tok
            # print(gen_text)
            # ori_text = single_lines[step]
            history_all.append(gen_text)
        print(history_all)
        results.append('\n'.join(history_all))
            
    
    data_file_path = os.path.join(args.save_samples_path,'zhouwenwang_turn_new_unsed_64.txt')
    io.open(data_file_path,'w').write('\n\n'.join(results))
    return data_file_path



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
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    config = RoFormerConfig.from_pretrained(args.model_path)
    model = RoFormerModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储
    # get_machine_metric_datas(model,args,tokenizer)
    generate_human_check_datas(model,args,tokenizer)
    


if __name__ == '__main__':
    main()
