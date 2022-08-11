

# CROSSTALK-GENERATION

:rocket::rocket::rocket::rocket::rocket::rocket:



:tada: The largest Chinese open source crosstalk dataset so far (collected from the Internet, books, and other open source projects)

:tada: Data types include stand-up crosstalk, paired-up crosstalk, group crosstalk, and sketches

:tada:Since this study is only for cross talk, the training set provided only includes cross talk. For other types, please refer to the **Data Description-Initial Data** module. 



------


- [Quick Start](#quick-start)
- [Data Description](#data-description)
- [Model description](#model-description)
- [Machine Metrics](#machine-metrics)
- [innovative Metrics](#innovative-metrics)
- [Human Scoring Metrics](#human-scoring-metrics)
- [Correlation coefficient](#correlation-coefficient-between-human-metrics-and-machine-metrics)
- [Generate examples](#generate-examples)



------
#### Quick Start：


![image-20220417153303002](https://github.com/anonNo2/crosstalk-generation/blob/main/img/container_server.png)


| model-type    | docker                   | file(googleDriver) | file(BaiduDriver)                                             |
| ------- | ----------------------------- | ---------------- | ------------------------------------------------------------ |
| T5      | anon2010/crosstalk-gen-env:t5 | [download](https://drive.google.com/file/d/1XCrhKk1D7CAkeYjYCtwBiRF0kAieEbsr/view?usp=sharing)             | [download](https://pan.baidu.com/s/1Wv73_Io7OyTUJZbkdDF9Iw?pwd=t8i9) |
| GPT     | ...                           | ...              | ...                                                          |
| Pangu-a | ...                           | ...              | ...                                                          |
| CPM     | ...                           | ...              | ...                                                          |

To be continued...

**Steps for usage：**

```
1.Download the relevant images and files first
2.Start the model image, take t5 as an example here
docker run -dp 32488:8080 --memory=3G -v <MODEL_RES>:/model_res anon2010/crosstalk-gen-env:t5
3.Access the host port

```

<MODEL_RES> : Model files decompression path

--memory：Specify the maximum memory limit, too small may cause failure to start





#### Data Description：

The data is divided into two parts: initial data and training data.

**initial data**：Including sketches crawled from the Internet, single performing, dual performing, group performing and Ketch comedy .



```
src/
----common_data/
--------meta.json                    metadata for full data
--------CompleteMetaExportData.zip   full data

```



| Type                                     | Number |
| -------------------------------------------- | ---- |
| single performing                                 | 168  |
| dual performing                                 | 3685 |
| group performing                                 | 256  |
| Ketch comedy                                     | 5222 |
| plain dialogue texts(extract from which text above) | 4859 |
| full data                                   | 9331 |



|               | Number     |
| --------------------- | -------- |
| Total words          | 16481376 |
| Number of utterances          | 663305   |
| Number of long utterances | 8717     |
| Number of short utterances   | 446756   |
| Median word numbers of utterances            | 16       |
| Mean utterances per script        | 71       |



**metadata format description：**

```
{
		"isAllDialog":true,                                       #Whether it is a pure dialogue format
		"charSize":526,                                           #Word count in this scripts
		"filePath":"u399dy/蔡少芬李菁相声《学说普通话》台词完整版",     #relative path
		"roles":[                                                 #roles
			"李菁:",
			"蔡少芬:"
		],
		"sentenceSize":25,                                        #utterances size
		"source":"https://www.399dy.com/xiangsheng/11176.html",   #source
		"idx":28,                                                 #index
		"title":"蔡少芬李菁相声《学说普通话》台词完整版",               #title
		"type":"对口相声"                                          #type
	},
```

**In-text formatting example：**

```
甲:现在的商业都讲究实事求是，公约上写着:“百问不烦，百拿不厌。”
乙:那是呀!
甲:旧社会做买卖就不一样，宣传得蛮好，实际净骗人，门口都写两块大牌子。
乙:写的什么?
甲:写着“货真价实，公平交易”。
乙:写得蛮好看!
甲:实际是货不真，价不实，大秤买，小秤卖，想尽一切办法多赚钱。大买卖讲究宣传，门口弄一份儿洋鼓洋号吹吹打打。还有的在电台登广告。广告都这样。
乙:您给学学。
甲:“各位先生，各位女士，早晨起来您不喝茶吗?您要想喝好茶叶的话，报告您一个好消息，××茶庄备有专人到南省产茶名区，采办各种红绿花茶，加花熏制，西湖龙井，铁叶儿大方，清香适口，气味芬芳，馈送亲友最为相宜。他家的地址:××大街往北路西一百七十三号，电话三局六二九四号。”
乙:是这样儿!
甲:这是那时候的大买卖。小买卖儿可报不起，做一天买卖连本带利将够一家子吃饭的，花不起广告费呀。像卖烤地瓜的也这么登广告就不行了。
乙:是吗?
甲:那稿子念起来也不受听啊!不信我给您念念。
乙:好。
```



------




**Train data**：The 2948 dialogues were cleaned and screened from the initial data, the validation set was 368 dialogues selected from the initial data, and the test set was 10 rounds of dialogue among the 50 dialogues selected from the initial data.Opened in project.(src/common_data)



```
src/
----common_data/
--------train_raw.zip           training corpus in original format（Contains meta information）
--------dev_raw.zip             Verification corpus in original format（Contains meta information）
--------train_pure_text.txt     Training corpus in plain text format
--------dev_pure_text.txt       Verification corpus in plain text format
--------test_filter_50x20.txt   test corpus（Machine index calculation and manual evaluation are based on this data）
```




| （data） | Number of scripts | number of utterances   | word count    |
| ------------ | ---- | ------ | ------- |
| Training corpus       | 2948 | 173194 | 3184664 |
| Verification       | 368  | 41482  | 905201  |
| test corpus       | 50   | 1000   | 19268   |


------


#### Model description：

finetune：

GPT,T5,T5-small,unilm,rnn,GPT3



zero-shot inference：

CPM-large,GPT3,Pangu-a,zhouwenwang

------



#### Machine Metrics



For details of the data files and benchmarking methods used for machine index scoring, see [Machine Metrics](https://github.com/anonNo2/crosstalk-generation/blob/main/eval_data/machine_eval/README.md) 



|                    | bleu_1    | bleu_2   | bleu_3   | bleu_4   | gleu     | rouge_1   | rouge_2  | rouge_L   | distinct_1 | distinct_2 |
| ------------------ | --------- | -------- | -------- | -------- | -------- | --------- | -------- | --------- | ---------- | ---------- |
| GPT-ep50           | 10.04     | 3.69     | 1.53     | 0.7      | 2.75     | 15.28     | 1.78     | 13.7      | 6.89       | 37.39      |
| T5-pesg-ep15       | 11.75     | 5.58     | 3.13     | 1.77     | 3.94     | 20.8      | 4.98     | 19.25     | 9.02       | 42.68      |
| Small_T5-pesg-ep95 | 11.71     | 5.39     | 2.93     | 1.67     | 3.64     | 19.98     | 4.37     | 18.61     | 8.08       | 36.38      |
| CPM_large          | 7.94      | 2.87     | 1.19     | 0.5      | 1.68     | 9.88      | 1.28     | 8.83      | 5.82       | 34.43      |
| UNILM_ep45         | 8.88      | 4.32     | 2.47     | 1.41     | 3.36     | 20.22     | 4.91     | 18.98     | 7.53       | 29.90      |
| RNN                | 11.77     | 4.02     | 1.47     | 0.57     | 2.49     | 17.25     | 2.13     | 15.94     | 4.73       | 16.23      |
| GPT3-base-Davinci  | **14.68** | **7.45** | **4.44** | **2.77** | 5.13 | 22.25 | **5.65** | 20.03     | 8.43       | 40.7       |
| GPT3-ft200-Davinci | 9.66      | 4.89     | 3.01     | 1.92     | 4.66     | 21.79     | 5.5      | 20.22 | **9.725**  | 43.15      |
| GPT3-ft1000-Davinci | 13.39      | 6.86     | 4.14     | 2.55     | **5.18**     | **22.83**     | 5.68      | **20.73** | 9.55  | 45.56      |
| Panggu-a           | 6.42      | 2.09     | 0.83     | 0.37     | 1.31     | 7         | 0.75     | 6.14      | 8.25       | 50.98      |
| zhouwenwang        | 7.33      | 2.26     | 0.9      | 0.4      | 1.81     | 10.41     | 1.01     | 8.61      | 9.72       | **53.53**  |

*PS:The reason why GPT3-ft1000 is not included in the paper is because we only select one of the similar models when manually marking. In the previous GPT3-finetune manual comparison, the manual score of ft200 is higher.*

|  | General quality | Humor | Coherence | Ethically-risky flag |
| ---------------------- | ------------- | ----------------- | --------------- | ----------- |
| GPT3-ft200-Davinci               | 145           | 115               | 43              | 2           |
| GPT3-ft1000-Davinci           | 136           | 109               | 40              | 2           |

**some small discoveries：**

1.GPT3-davinci's machine metrics are significantly higher than any other generation method。

2.Most large models zero-shot inference significantly worse direct inference than finetune.

3.After GPT3 goes through finetune, the machine index drops, but in the following manual scoring, it will get a higher score.(Higher machine indicators are not necessarily more in line with human reading habits?)


------



#### innovative Metrics：



|      | zhouwenwang | unilm   | t5      | smallt5-95 | rnn     | pangu_a | gpt3_finetune | gpt3_base | gpt_ep50 | CPM_large |
| ---- | ----------- | ------- | ------- | ---------- | ------- | ------- | ------------- | --------- | -------- | --------- |
| 1    | 0.0695      | 0       | 0       | 0          | 0       | 0.00754 | 0             | 0.00293   | 0        | 0.04777   |
| 2    | 0.23223     | 0.07405 | 0.03709 | 0.02538    | 0.01244 | 0.10615 | 0.02337       | 0.028     | 0.00826  | 0.22447   |
| 3    | 0.58549     | 0.37475 | 0.22624 | 0.19325    | 0.10785 | 0.43833 | 0.16998       | 0.18063   | 0.13619  | 0.53327   |
| 4    | 0.83795     | 0.68884 | 0.51332 | 0.45936    | 0.32618 | 0.73634 | 0.4297        | 0.42368   | 0.43385  | 0.78466   |
| 5    | 0.94343     | 0.86874 | 0.75219 | 0.68772    | 0.62232 | 0.91339 | 0.67156       | 0.64273   | 0.73764  | 0.91419   |
| 6    | 0.97811     | 0.94026 | 0.88568 | 0.82587    | 0.84941 | 0.97563 | 0.81931       | 0.76029   | 0.91134  | 0.9615    |
| 7    | 0.98968     | 0.96837 | 0.94607 | 0.90037    | 0.94703 | 0.99402 | 0.88814       | 0.80665   | 0.97395  | 0.98139   |
| 8    | 0.99347     | 0.98052 | 0.97288 | 0.94256    | 0.98062 | 0.9987  | 0.91946       | 0.82957   | 0.99267  | 0.99015   |
| 9    | 0.99544     | 0.98527 | 0.98457 | 0.9669     | 0.99051 | 0.99972 | 0.93244       | 0.84394   | 0.99757  | 0.99252   |
| 10   | 0.99711     | 0.98799 | 0.99007 | 0.98092    | 0.99472 | 1       | 0.93793       | 0.85047   | 0.9993   | 0.99393   |



![image-20220417153303002](https://github.com/anonNo2/crosstalk-generation/blob/main/img/image-20220417153303002.png)



**some small discoveries：**

1.zhouwenwang，CPM，pangu-a（zero-shot inference）is more innovative.

2.In the finetune model, unilm and T5 are relatively more innovative.

3.Models with higher innovative evaluations are often generated more randomly. 

------



#### Human Scoring Metrics：

The numbers in parentheses of the column names are human scoring of the real data

The combined score and humor score are capped at 750

Fluency and Discrimination Scores are capped at 150



For details of the data files used for manual scoring and the scoring method, see [manual scoring](https://github.com/anonNo2/crosstalk-generation/blob/main/eval_data/human_eval/README.md) 



Test Design： Here we will infer a total of 10 pre-trained models and large models (excluding T5-small, because the effect is not as good as T5, plus the original data) for the first 10 sentences of the 50 segments of the test set to infer the next 10 sentences，A total of 30 subjects with different genders, occupations, ages, and beliefs were found and asked to rate the samples generated by the model. The comprehensive score and humor score were scored on a 5-point scale, and the fluency and discrimination were on a 1-point scale (1 if yes, otherwise 0).

|  | General quality(528) | Humor（519） | Coherence(143) | Ethically-risky flag(3) |
| ---------------------- | ------------- | ----------------- | --------------- | ----------- |
| GPT-ep50               | 225           | 256               | 59              | 2           |
| T5-pesg-ep15           | 270           | 296               | 76              | 7           |
| CPM_large              | 213           | 240               | 60              | **34**      |
| UNILM_ep45             | 276           | 301               | 84              | 2           |
| RNN                    | 217           | 242               | 41              | 4           |
| GPT3-base-Davinci      | 322           | 325               | 98              | 5           |
| GPT3-ft200-Davinci     | **341**       | **353**           | **106**         | 2           |
| Panggu-a               | 230           | 257               | 63              | 4           |
| zhouwenwang            | 184           | 191               | 28              | 8           |



**some small discoveries：**

1.Although GPT3's machine indicators have declined after finetune, all dimensions of manual scoring are ranked first.Machine metrics don't fully assess how well a generation works.

2.Compared with GPT and RNN, the generation quality of model structures such as UNILM and T5 will be significantly improved.

------



#### Correlation coefficient between human Metrics and machine Metrics：

![pearsonr-Correlation](https://github.com/anonNo2/crosstalk-generation/blob/main/img/pearsonr-Correlation%20coefficient.png)

![pearsonr-Correlation](https://github.com/anonNo2/crosstalk-generation/blob/main/img/spearman-Correlation%20coefficient.png)

------



#### Generate examples：

Short stories generated by GPT3 : [Chinese](https://github.com/anonNo2/crosstalk-generation/blob/main/GPT3-Generate-Samples/Short-Sample.zh-CN.md) [English](https://github.com/anonNo2/crosstalk-generation/blob/main/GPT3-Generate-Samples/Short-Sample.en-US.md)  
Long stories generated by GPT3 :[Chinese](https://github.com/anonNo2/crosstalk-generation/blob/main/GPT3-Generate-Samples/Long-Sample.zh-CN.md) [English](https://github.com/anonNo2/crosstalk-generation/blob/main/GPT3-Generate-Samples/Long-Sample.en-US.md)

**Interactive generation**

```
demo 1:

user:大家好
chatbot:大家好
user:今天我们两个和大家说段相声
chatbot:对，这是全国有名的主持人曹云金
user:不敢不敢
chatbot:首先我们两个人说的头一回合作
user:是，第一回
chatbot:我们两个人说相声，有名
user:全国老少爷们都知道
chatbot:我一个人说相声，有名啊
user:感情您这是要夸自己呀
chatbot:谢我干吗呀
user:谢您夸自个儿的时候带上我了
chatbot:对您有好处啊


demo 2:

user:谢谢大家热烈的掌声
chatbot:谢大伙儿
user:我们哥俩儿今天来给大家说段相声
chatbot:对，这个相声主要讲究的是说、学。
user:还有逗唱
chatbot:相声主要讲究说、学表演
user:你还和我杠上了是么
chatbot:怎么就我不是一个人
user:那你是什么呀
chatbot:我不是一个人

demo 3:

user:大家好呀，掌声很热烈
chatbot:是
user:我们哥俩今天来给大伙说短相声
chatbot:别说了
user:啊，为什么呀
chatbot:这个相声是我最重要的朋友?怕您各位不认识，有可能还在暗地里面呢
user:什么跟什么呀这是
chatbot:怕您各位不认识我们
user:那我们先来做一下自我介绍
chatbot:做个说明
user:我是anon
chatbot:我叫韩雪松，谢大家听个话

```

**script generation**

```
demo 1
我们首先看一下，其他剧场相声的报价。
那我们来听一听。
国家大剧院一千五百块，农业展览馆二千块，北京动物园五千块。
跟猴儿一块说相声啊?
那么我们今天的相声卖多少钱呢?
卖多少钱哪?
八毛钱，比其他剧场足足便宜了三百块呀!
这都什么账头啊这是?
赶快拨打我们的销售热线，二百五加三八加十三点加二。
全是傻数啊。
========================以下为生成内容===========================
你拨了?
拨了啊。
你说什么?
我说，我是一个人，一个人来。
这么小的数目，我们还是先报给你吧。
好啊。
一共十五块钱，您有现金吗?
有呀!您给我十五块钱。
不用给十五块钱了，跟我来吧!


demo 2
谢谢大伙儿，谢谢大伙儿。
感谢您各位。
这次啊回到北京说相声，我这心里是特别的高兴。
诚惶诚恐。
尤其见着下面几位还是我们的老街坊。
老街坊。
看见您那我就想起我小时候住的那骆驼湾胡同来。
骆驼湾那儿。
可不是嘛，过去有个北京的小曲。
叫什么呀?
========================以下为生成内容===========================
叫做《满江红》。
《满江红》。
这首歌是我们老街坊常唱的。
小时候我们还在上海呢，我们小时候也常听到这首歌。
小时候听到这首歌，我们都会跟着一起唱。
那个时候，你说的是北京人，你说的是北京人，北京人都会一起唱的。
对对对对对对！
不要只有北京人会一起唱，全国各地的人都会一起唱。
全国各地的人都会一起唱。

demo 3
人人都有爱好。
是
有爱好听歌的，有爱好看书的，有爱好玩游戏的，你的爱好就很好，很高雅。
什么啊
他爱踢毽子
对
踢毽子好啊
是么
促进血液循环，血液循环有规律对身体很好
有这么大用处?
========================以下为生成内容===========================
对啊
那是
但是也有高雅之风
哦?
有时候踢得动人
是吗
尤其是在春晚上
踢得动人
是
我跟朋友们说
```

------



####  Example of generating different models in the same context

**demo 1**

```


======================================
原始篇章：

谢谢大家。
唉。
刚才是郭老师带着他的爱徒，岳云鹏。
嗯。
两个人合说一段歪唱太平歌词。
嗯。
2700多位观众啊，乐的都累了。
是。
把我们换上来，让您休息休息。
对。
-----------------------------------本线之前是context,之后为基于context的剧本生成
岳云鹏，有人缘。
嗯。
长的也漂亮嘛。
好啊!
浓眉小眼，大屁股圆脸。
您还什么都见过。
这是他的一个艺术特点。
这是艺术特点。
郭德纲，大家更熟悉了。
嗯。
======================================
UNILM:

甲:谢谢大家。
乙:唉。
甲:刚才是郭老师带着他的爱徒，岳云鹏。
乙:嗯。
甲:两个人合说一段歪唱太平歌词。
乙:嗯。
甲:2700多位观众啊，乐的都累了。
乙:是。
甲:把我们换上来，让您休息休息。
乙:对。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:我们两个人合作这么长时间。
乙:嗯。
甲:刚才主持人叫郭麒麟。
乙:哦。
甲:他也给大家表演一段传统相声。
乙:对。
甲:后边那位老师。
乙:嗯。
甲:就是孙越老师。
乙:对。

======================================
GPT

甲:谢谢大家。
乙:唉。
甲:刚才是郭老师带着他的爱徒，岳云鹏。
乙:嗯。
甲:两个人合说一段歪唱太平歌词。
乙:嗯。
甲:2700多位观众啊，乐的都累了。
乙:是。
甲:把我们换上来，让您休息休息。
乙:对。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:我们这一场换将上来。
乙:我跟您说这正经的咱们。
甲:是。
乙:由于压力比较大。
甲:哦。
乙:正经的多了。
甲:嗯。
乙:四门功课，绕口令您怎么样?
甲:我会数数来宝，其实学这东西好啊。
乙:学习数来宝。

======================================
GPT3

甲:谢谢大家。
乙:唉。
甲:刚才是郭老师带着他的爱徒，岳云鹏。
乙:嗯。
甲:两个人合说一段歪唱太平歌词。
乙:嗯。
甲:2700多位观众啊，乐的都累了。
乙:是。
甲:把我们换上来，让您休息休息。
乙:对。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:现在我们请来了一位新人，让大家给他欢呼。
乙:嗯。
甲:这个新人是谁呢？
乙:嗯。
甲:这个新人是谁呢？
乙:嗯。
甲:刚才我们在这里说了一句话，不知道大家听懂了没有？
乙:对。你说的是什么呢？
甲:我们在这里说了一句话——“好的”，不知道大家听懂了没有？
乙:嗯。我听懂了。

======================================
Seq2Seq

甲:谢谢大家。
乙:唉。
甲:刚才是郭老师带着他的爱徒，岳云鹏。
乙:嗯。
甲:两个人合说一段歪唱太平歌词。
乙:嗯。
甲:2700多位观众啊，乐的都累了。
乙:是。
甲:把我们换上来，让您休息休息。
乙:对。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:我们俩是两个人。
乙:嗯。
甲:咱们俩演员一块栾云平方休息，先生在艺术家里边站着。
乙:对。
甲:相声界的艺术家。
乙:哎。
甲:京剧水平高峰。
乙:是。
甲:今天我们哥俩表演艺术家。
乙:是。

======================================



```

**demo 2**

```




======================================
原始篇章：

甲:相声演员李伟建。
乙:刘洪沂。
甲:今天呢我们给您说段相声。
乙:刘老师。
甲:啊。
乙:先问问您。
甲:啊。
乙:中国这么多传统的节日，您最爱过哪个节啊?
甲:爱过哪个节?
乙:啊。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:我最爱过的是那个……
乙:您这么羞涩干嘛呀。
甲:那我最爱过的是春节。
乙:春节。
甲:啊。
乙:为什么呀?
甲:你想啊，新年伊始，万象更新，中国的传统节日大伙都这么过嘛。
乙:您今年多大岁数?
甲:我今年六十五岁。
乙:难怪。
======================================
UNILM:

甲:相声演员李伟建。
乙:刘洪沂。
甲:今天呢我们给您说段相声。
乙:刘老师。
甲:啊。
乙:先问问您。
甲:啊。
乙:中国这么多传统的节日，您最爱过哪个节啊?
甲:爱过哪个节?
乙:啊。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:寒食节。
乙:对呀。
甲:中国有句话叫"吃粽子怕死"。
乙:哦。
甲:而且呢也要过。
乙:是啊。
甲:尤其是端午节。
乙:诶。
甲:吃粽子。
乙:吃粽子怕死?

======================================
GPT

甲:相声演员李伟建。
乙:刘洪沂。
甲:今天呢我们给您说段相声。
乙:刘老师。
甲:啊。
乙:先问问您。
甲:啊。
乙:中国这么多传统的节日，您最爱过哪个节啊?
甲:爱过哪个节?
乙:啊。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:爱过传统节日。
乙:这回你打算表演什么节目?
甲:我找一下好节目啊，让您开眼。
乙:等会儿。
甲:民主席是谁啊?
乙:哈利波特。
甲:好啊。
乙:他在美国还给安排了。
甲:怎么没有选项啊?
乙:你们也去呗!

======================================
GPT3

甲:相声演员李伟建。
乙:刘洪沂。
甲:今天呢我们给您说段相声。
乙:刘老师。
甲:啊。
乙:先问问您。
甲:啊。
乙:中国这么多传统的节日，您最爱过哪个节啊?
甲:爱过哪个节?
乙:啊。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:嗯。
乙:啊。
甲:爱过哪个节?
乙:啊。
甲:我觉得，爱过的节日，是农历新年。
乙:哦。
甲:因为我们中国人，最重要的一个节日，就是农历新年。
乙:那么您呢，您最爱过的一个人呢?
甲:我最爱过的一个人?
乙:啊。
======================================
Seq2Seq

甲:相声演员李伟建。
乙:刘洪沂。
甲:今天呢我们给您说段相声。
乙:刘老师。
甲:啊。
乙:先问问您。
甲:啊。
乙:中国这么多传统的节日，您最爱过哪个节啊?
甲:爱过哪个节?
乙:啊。
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:喜欢听节日。
乙:对。
甲:喜欢相声。
乙:喜欢。
甲:喜欢健美。
乙:喜欢。
甲:喜欢看。
乙:喜欢看。
甲:喜欢看。
乙:喜欢看。

======================================


```

**demo 3**

```

======================================
原始篇章：

甲:今天呢我给大家表演相声。
乙:像您这个相声演员都讲究什么。
甲:讲究说学逗唱。
乙:说学逗唱。
甲:对。
乙:唱你能唱什么。
甲:唱我们各地方的地方戏，评戏，梆子，京剧，尤其我们相声演员对京剧非常有研究。
乙:这么说您对京戏非常有研究。
甲:那是啊。
乙:那您会整出戏吗?
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:经常跟这个京剧演员在一块唱。
乙:是吗?
甲:那是。
乙:那行，今儿个我帮帮您。
甲:帮，甭帮，经不住帮。
乙:我打算帮您唱一出。
甲:您帮我唱一出戏。
乙:啊。
甲:没问题。
乙:那咱俩唱一出。
======================================
UNILM:

甲:今天呢我给大家表演相声。
乙:像您这个相声演员都讲究什么。
甲:讲究说学逗唱。
乙:说学逗唱。
甲:对。
乙:唱你能唱什么。
甲:唱我们各地方的地方戏，评戏，梆子，京剧，尤其我们相声演员对京剧非常有研究。
乙:这么说您对京戏非常有研究。
甲:那是啊。
乙:那您会整出戏吗?
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:我会。
乙:整出戏你唱什么呀?
甲:整出戏叫<<西河调>。
乙:哎哟喂。
甲:怎么着?
乙:打算学谁呢?
甲:我呀。
乙:打算学谁呀?
甲:我呀。
乙:打算学谁呀?

======================================
GPT

甲:今天呢我给大家表演相声。
乙:像您这个相声演员都讲究什么。
甲:讲究说学逗唱。
乙:说学逗唱。
甲:对。
乙:唱你能唱什么。
甲:唱我们各地方的地方戏，评戏，梆子，京剧，尤其我们相声演员对京剧非常有研究。
乙:这么说您对京戏非常有研究。
甲:那是啊。
乙:那您会整出戏吗?
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:不会我干嘛你说废话，我还没学会。
乙:是啊。
甲:要不咱来点个别的。
乙:看出什么呀。
甲:来也没有唱过，我唱过什么京剧。
乙:那可不一定。
甲:您能找一个能人吗?
乙:哪儿的事情啊。
甲:您跟我唱就行了，来这苏三。
乙:苏三?
======================================
GPT3

甲:今天呢我给大家表演相声。
乙:像您这个相声演员都讲究什么。
甲:讲究说学逗唱。
乙:说学逗唱。
甲:对。
乙:唱你能唱什么。
甲:唱我们各地方的地方戏，评戏，梆子，京剧，尤其我们相声演员对京剧非常有研究。
乙:这么说您对京戏非常有研究。
甲:那是啊。
乙:那您会整出戏吗?
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:我整出戏。
乙:那您整出什么戏?
甲:啊，我整出《绣花大盗》。
乙:《绣花大盗》。
甲:对，《绣花大盗》。
乙:那您整出什么地方的戏?
甲:啊，这个是关于一个叫王家卫的人。
乙:王家卫。
甲:对，王家卫，这个人有一天他去看电影，看到电影里的女主角带了一个小鸟，小鸟叫做小白兔，小白兔是一只很懒惰的小鸟，因此王家卫急于想要这只小鸟。
======================================
Seq2Seq

甲:今天呢我给大家表演相声。
乙:像您这个相声演员都讲究什么。
甲:讲究说学逗唱。
乙:说学逗唱。
甲:对。
乙:唱你能唱什么。
甲:唱我们各地方的地方戏，评戏，梆子，京剧，尤其我们相声演员对京剧非常有研究。
乙:这么说您对京戏非常有研究。
甲:那是啊。
乙:那您会整出戏吗?
-----------------------------------本线之前是context,之后为基于context的剧本生成
甲:戏剧不少。
乙:还没有?
甲:哎。
乙:您怎么回事?
甲:我也没耽误工夫。
乙:您别提这个忙活动。
甲:您这话说得很丰富。
乙:噢。
甲:您看我们这些个社会最近常用的。
乙:噢，您可以参加各种曲艺协会来解放的。

======================================




```
