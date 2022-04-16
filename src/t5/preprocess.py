from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from transformers import BertTokenizerFast
import argparse
import pandas as pd
import pickle,os
import jieba.analyse
from tqdm import tqdm

import logging,json
import numpy as np

from t5_tokenizer import T5PegasusTokenizer


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def reduce_data(data,logger,args,file_name):

    # 初始化tokenizer
    tokenizer = T5PegasusTokenizer(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    sep_tok = tokenizer.sep_token
    cls_tok = tokenizer.cls_token
    
    # 需要区分linux和windows环境下的换行符
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialogue in dataset".format(len(train_data)))
    save_file_path = os.path.join(args.save_path,file_name + '.pkl')
    save_content_path = os.path.join(args.save_path,file_name + '_raw.txt')
    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    dialogue_len = []  # 记录所有对话tokenize之后的长度，用于统计中位数与均值
    dialogue_list = []
    utterance_list = []


    

    
    for index, dialogue in enumerate(tqdm(train_data)):
        if "\r\n" in data:
            utterances = dialogue.split("\r\n")
        else:
            utterances = dialogue.split("\n")

        reduce_max_len_utterances = []
        reduce_max_len_utterances_toks = []
        reduce_max_len_utterances_ids = []

        for idx in range(len(utterances)):
            candi_sent = utterances[idx]
            if len(candi_sent) > args.sent_max_len:
                candi_sent = candi_sent[:args.sent_max_len]
            reduce_max_len_utterances.append(candi_sent)
            reduce_max_len_utterances_toks.append(tokenizer.tokenize(candi_sent) + [sep_tok])
            reduce_max_len_utterances_ids.append(tokenizer.encode(candi_sent, add_special_tokens=False) + [sep_id])
        
        for step in range(1,len(reduce_max_len_utterances)):

            inputs_text_ids = reduce_max_len_utterances_ids[:step]
            inputs_text_toks = reduce_max_len_utterances_toks[:step]
            
            labels_ids = [cls_id] + reduce_max_len_utterances_ids[step] 
            labels_tok = [cls_tok] + reduce_max_len_utterances_toks[step]

            history_start_index = 1
            filter_history_sent = []
            filter_history_sent_ids = []
            input_ids = [cls_id]  # 每个dialogue以[CLS]开头
            input_toks = [cls_tok]
            # 逻辑是先从最后一位往前加句子，加下一句如果总数超了max_len就停止
            # （ps） gpt的 generate_text_by_input 方法里history回溯那里写错了，不能从history的头部开始回溯，应该从尾部，否则我们想要的是 BCD->E,会得到ABC->E
            for rev_idx in range(len(inputs_text_ids)-1,-1,-1):
                this_turn_toks = inputs_text_toks[rev_idx]
                
                this_turn_ids = inputs_text_ids[rev_idx]
                
                if history_start_index + len(this_turn_ids)  > args.para_max_len:
                    break
                filter_history_sent.append(this_turn_toks)
                filter_history_sent_ids.append(this_turn_ids)
                history_start_index += len(this_turn_ids)
            
            filter_history_sent.reverse()
            filter_history_sent_ids.reverse()
            for his_idx in range(len(filter_history_sent)):
                input_ids.extend(filter_history_sent_ids[his_idx])
                input_toks.extend(filter_history_sent[his_idx])
            
            
            content_line = json.dumps({'src':input_toks,'tgt':labels_tok},ensure_ascii=False)
            ids_line = json.dumps({'src':input_ids,'tgt':labels_ids},ensure_ascii=False)
        
            dialogue_len.append(len(input_ids))

            utterance_list.append(content_line + '\n')
            dialogue_list.append(ids_line + '\n')

        
    len_mean = np.mean(dialogue_len)
    len_median = np.median(dialogue_len)
    len_max = np.max(dialogue_len)

    with open(save_content_path,'w') as f:
        f.write(''.join(utterance_list))
    
    with open(save_file_path, "wb") as f:
        pickle.dump(dialogue_list, f)
    logger.info("finish preprocessing {} data,the result is stored in {}".format(file_name,save_file_path))
    logger.info("mean of dialogue len:{},median of dialogue len:{},max len:{}".format(len_mean, len_median, len_max))


def preprocess():
    """
    对原始语料进行tokenize，将每段对话处理成如下形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='/data1/anon/crosstalk-generation/pretrain_model/t5_pegasus_torch/vocab.txt', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--para_max_len', default=256, type=int, required=False, help='单条训练语料最大长度')
    parser.add_argument('--sent_max_len', default=64, type=int, required=False, help='单条句子最大长度')
    parser.add_argument('--data_base', default='/data1/anon/crosstalk-datasets/data_resource/formal_data', type=str, required=False, help='数据文件存储位置')
    parser.add_argument('--save_base', default='/data1/anon/crosstalk-generation/src/t5/data/', type=str, required=False, help='tokenize的训练数据集')
    args = parser.parse_args()

    args.save_path = os.path.join(args.save_base,f'p{args.para_max_len}-s{args.sent_max_len}')
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


        # 初始化日志对象
    logger = create_logger(os.path.join(args.save_path,'preprocess.log'))
    # 读取训练数据集
    for file_name in ['train','dev','test']:
        with open(os.path.join(args.data_base,file_name + '.txt'), 'rb') as f:
            data = f.read().decode("utf-8")
        reduce_data(data,logger,args,file_name)
    

    


if __name__ == '__main__':
    preprocess()

