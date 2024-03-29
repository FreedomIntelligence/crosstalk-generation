# Machine Scoring Details



#### 1.Machine Scoring Data File Description：

```
data:
|-------gpt3_base_davinci_metric.json							gpt3 davinci Evaluation data
|-------gpt3_davinci_ft_metric.json               gpt3 davinci finetune Evaluation data
|-------machine_metric_data_CPM_large.json        CPM large Evaluation data
|-------machine_metric_data_gpt_ep50.json         GPT finetune Evaluation data
|-------machine_metric_data_pangu_a.json          pangu-a Evaluation data
|-------machine_metric_data_smallt5-95.json       small-t5 finetune Evaluation data
|-------machine_metric_data_unilm.json            UNILM finetune Evaluation data
|-------machine_metric_data_zhouwenwang.json      zhouwenwang finetune Evaluation data
|-------machine_metric_t5_pesg_ep15.json          T5-pesg finetune Evaluation data
|-------rnn_predic.json                            rnn Evaluation data
```

#### 2.json结构说明

```
    {
        "ori": "是啊，好久不见。不过我听说你最近挺忙的。",         original sentence
        "gen": "我在哪里,怎么还不来"                           AB-style generating sentences
    },
```

The evaluation data of the machine running points are all generated in the way of AB sentences






#### 3.Model source：

GPT3 : https://beta.openai.com/

CPM：https://huggingface.co/TsinghuaAI/CPM-Generate

GPT：https://huggingface.co/thu-coai/CDial-GPT_LCCC-base

Pangu-a：https://huggingface.co/imone/pangu_2_6B

T5-small：https://huggingface.co/imxly/t5-pegasus-small

T5：https://huggingface.co/imxly/t5-pegasus

UNILM：https://github.com/YunwenTechnology/Unilm

ZhouWenWang：https://huggingface.co/IDEA-CCNL/Zhouwenwang-Unified-1.3B



#### 4.Evaluation Metrics：


|                    | bleu_1    | bleu_2   | bleu_3   | bleu_4   | gleu     | rouge_1   | rouge_2  | rouge_L   | distinct_1 | distinct_2 |
| ------------------ | --------- | -------- | -------- | -------- | -------- | --------- | -------- | --------- | ---------- | ---------- |
| GPT-ep50           | 10.04     | 3.69     | 1.53     | 0.7      | 2.75     | 15.28     | 1.78     | 13.7      | 6.89       | 37.39      |
| T5-pesg-ep15       | 11.75     | 5.58     | 3.13     | 1.77     | 3.94     | 20.8      | 4.98     | 19.25     | 9.02       | 42.68      |
| Small_T5-pesg-ep95 | 11.71     | 5.39     | 2.93     | 1.67     | 3.64     | 19.98     | 4.37     | 18.61     | 8.08       | 36.38      |
| CPM_large          | 7.94      | 2.87     | 1.19     | 0.5      | 1.68     | 9.88      | 1.28     | 8.83      | 5.82       | 34.43      |
| UNILM_ep45         | 8.88      | 4.32     | 2.47     | 1.41     | 3.36     | 20.22     | 4.91     | 18.98     | 7.53       | 29.90      |
| RNN                | 11.77     | 4.02     | 1.47     | 0.57     | 2.49     | 17.25     | 2.13     | 15.94     | 4.73       | 16.23      |
| GPT3-base-Davinci  | **14.68** | **7.45** | **4.44** | **2.77** | **5.13** | **22.25** | **5.65** | 20.03     | 8.43       | 40.7       |
| GPT3-ft200-Davinci | 9.66      | 4.89     | 3.01     | 1.92     | 4.66     | 21.79     | 5.5      | **20.22** | **9.725**  | 43.15      |
| Panggu-a           | 6.42      | 2.09     | 0.83     | 0.37     | 1.31     | 7         | 0.75     | 6.14      | 8.25       | 50.98      |
| zhouwenwang        | 7.33      | 2.26     | 0.9      | 0.4      | 1.81     | 10.41     | 1.01     | 8.61      | 9.72       | **53.53**  |

The data results can be obtained through machine_metrics.py in the same directory