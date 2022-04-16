import argparse
from lib2to3.pgen2 import token
import math
import time
from urllib import response
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.nn import DataParallel
import transformers
import pickle
import sys,random,io,json
from generate_eval_data import get_machine_metric_datas
from metrics import cal_metrics
from pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split
from data_parallel import BalancedDataParallel
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from dataset import MyDataset
from t5_tokenizer import T5PegasusTokenizer



def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--data_dir', default='/data1/anon/crosstalk-generation/src/t5/data/p256-s64', type=str, required=False, help='数据基础路径路径')
    parser.add_argument('--max_len', default=256, type=int, required=False, help='训练时，输入数据的最大长度')

    
    parser.add_argument('--log', default=True, help="是否记录日志")
    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')
    
    parser.add_argument('--epochs', default=100, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--save_epochs', default=5, type=int, required=False, help='几个epoch保存一次')
    parser.add_argument('--batch_size', default=24, type=int, required=False, help='训练的batch size')
    parser.add_argument('--gpu0_bsz', default=2, type=int, required=False, help='0号卡的batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='衰减率')
    parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--save_model_base_path', default='/data1/anon/crosstalk-generation/trained_model_dir/t5-pegasus', type=str, required=False,
                        help='模型输出总路径')
    parser.add_argument('--pretrained_model', default='/data1/anon/crosstalk-generation/pretrain_model/t5_pegasus_torch', type=str, required=False,
                        help='预训练的模型的路径')
    parser.add_argument('--seed', type=int, default=1234, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_rate', type=float, default=0.1, help='warm up步数占总步数的比例')

    
    # 跑机器分所需参数(belu,gelu,rouge)
    parser.add_argument('--save_cal_metric', default=True, help="是否在保存时计算机器指标，汇总结果会放在 save_model_base_path 下")
    parser.add_argument('--min_limit_epoch', default=10, type=int, help="在哪个epoch后开始进行机器指标计算，(一先开始的n个epoch可能没有计算价值)")
    parser.add_argument('--test_filter_data', default="/data1/anon/crosstalk-generation/src/t5/data/p256-s64/test_filter_50x20.txt", type=str, required=False, help="数据基准文件，n个篇章，每个篇章20行")
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seq_max_len', type=int, default=256, help='篇章最大长度')
    parser.add_argument('--utterance_max_len', type=int, default=64, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--save_samples_path', default="/data1/anon/crosstalk-generation/src/t5/sample/", type=str, required=False, help="保存跑机器指标数据的文件路径及保存篇章级生成的文件路径")


    args = parser.parse_args()
    return args

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


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence([i[0] for i in batch], batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence([i[1] for i in batch], batch_first=True, padding_value=-100)
    return input_ids, labels


# def padding_batch(data_list, pad_id):
#     """
#     使用pad_id将data_list的每条数据，填充至data_list中最长的长度
#     :param data_list:
#     :param pad_id:
#     :return:
#     """
#     # 统计data_list中的最大长度
#     max_len = 0
#     for data in data_list:
#         max_len = max_len if max_len > len(data) else len(data)
#
#     # 对数据进行padding
#     new_data_list = []
#     for data in data_list:
#         new_data = data + [pad_id] * (max_len - len(data))
#         new_data_list.append(new_data)
#     return new_data_list


def load_dataset(logger, args):
    """
    加载训练集和验证集
    """
    logger.info("loading training dataset and validating dataset")
    train_path = os.path.join(args.data_dir,'train.pkl')
    valid_path = os.path.join(args.data_dir,'dev.pkl')
    test_path = os.path.join(args.data_dir,'test.pkl')

    with open(train_path, "rb") as f:
        train_input_list = pickle.load(f)
    with open(valid_path, "rb") as f:
        valid_input_list = pickle.load(f)
    with open(test_path, "rb") as f:
        test_input_list = pickle.load(f)


    train_dataset = MyDataset(train_input_list, args.max_len)
    val_dataset = MyDataset(valid_input_list, args.max_len)
    test_dataset = MyDataset(test_input_list, args.max_len)

    return train_dataset, val_dataset, test_dataset


def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
                epoch, args,tokenizer):
    model.train()
    device = args.device
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss的总和

    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                        batch_idx + 1, epoch + 1, loss.item() * args.gradient_accumulation_steps, batch_acc, scheduler.get_lr()))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    if (epoch+1) % args.save_epochs == 0:

        # save model
        logger.info('saving model for epoch {}'.format(epoch + 1))
        model_path = join(args.save_model_path, 'epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        logger.info(f'{epoch}/{args.epochs} saving model~')
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        if args.save_cal_metric and epoch >= args.min_limit_epoch:
            logger.info(f'we should calculate machine metrics in epoch {epoch}')
            # 生成metric的主要逻辑
            generate_file_path = get_machine_metric_datas(model_to_save,args,tokenizer,epoch)
            machine_metrics = cal_metrics(generate_file_path)
            io.open(os.path.join(model_path,'machine_metrics.json'),'w').write(json.dumps(machine_metrics,indent=4))
            


    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss



def validate_epoch(model, validate_dataloader, logger, epoch, args):
    logger.info("start validating")
    model.eval()
    device = args.device
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0

    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                
                total_loss += loss.item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}, ppl {}".format(epoch+1, epoch_mean_loss, np.exp(epoch_mean_loss)))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exception))
            raise exception


def train(model, logger, train_dataset, validate_dataset, args,tokenizer):
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_model_path)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_rate * t_total), num_training_steps=t_total
    )

    logger.info('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    for epoch in range(args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args,tokenizer=tokenizer)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader,
            logger=logger, epoch=epoch, args=args)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch + 1))
            model_path = join(args.save_model_path, 'min_ppl_model'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

        #  如果patience=0,则不进行early stopping
        if args.patience == 0:
            continue
        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))


def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def main():
    # 初始化参数
    args = set_args()

    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # 创建模型的输出目录
    args.save_model_path = os.path.join(args.save_model_base_path,f'ml-{args.max_len}-seed-{args.seed}')
    args.log_path = os.path.join(args.save_model_base_path,'train.logs')
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    # 创建日志对象
    logger = create_logger(args)
    logger.info('#' * 30)
    logger.info(args)
    logger.info('#' * 30)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    # 初始化tokenizer
    tokenizer = T5PegasusTokenizer(vocab_file=os.path.join(args.pretrained_model,'vocab.txt'), sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id
    

    # 创建模型
    model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)
    model = model.to(device)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # 并行训练模型
    if args.cuda and torch.cuda.device_count() > 1:
        
        if args.batch_size != args.gpu0_bsz:
            model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
        else:
            model = DataParallel(model).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset, val_dataset, test_dataset = load_dataset(logger, args)

    train(model, logger, train_dataset, val_dataset, args, tokenizer)


if __name__ == '__main__':
    main()
