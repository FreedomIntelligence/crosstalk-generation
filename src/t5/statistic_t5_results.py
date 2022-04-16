import argparse
import os,io,json
import pandas as pd
from generate_eval_data import get_machine_metric_datas
from metrics import cal_metrics

from t5_tokenizer import T5PegasusTokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer
base_path = '/data1/anon/crosstalk-generation/trained_model_dir/t5-small/ml-256-seed-1234'



def statistic_machine_metrics():
    
    start_ep = 10
    results = []
    while True:
        sub_dir = f'epoch{start_ep}'
        complete_dir = os.path.join(base_path,sub_dir)
        if not os.path.exists(complete_dir):
            break
        metric_file = os.path.join(complete_dir,'machine_metrics.json')
        metrics = json.loads(io.open(metric_file,'r').read())
        metrics['ep'] = start_ep
        start_ep += 5
        results.append(metrics)
    pd.DataFrame(results).to_excel('metrics_new_t5.xls')


def generate_machine_metrics(args):
    pretrain_dir = '/data1/anon/crosstalk-generation/pretrain_model/chinese_t5_pegasus_small'
    tokenizer = T5PegasusTokenizer(vocab_file=os.path.join(pretrain_dir,'vocab.txt'), sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    start_ep = 10
    while True:
        sub_dir = f'epoch{start_ep}'
        complete_dir = os.path.join(base_path,sub_dir)
        if not os.path.exists(complete_dir):
            break
        metric_file = os.path.join(complete_dir,'machine_metrics.json')
        model = MT5ForConditionalGeneration.from_pretrained(complete_dir)
        model.to('cuda')
        sample_file = get_machine_metric_datas(model,args,tokenizer,f'smallt5-{start_ep}')
        machine_metrics = cal_metrics(sample_file)
        io.open(metric_file,'w').write(json.dumps(machine_metrics,indent=4))
        print(f'start_ep {start_ep} is finish!')
        
        start_ep += 5
        
parser = argparse.ArgumentParser()
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
generate_machine_metrics(args)
statistic_machine_metrics()