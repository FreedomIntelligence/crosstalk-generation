# 

# GPT3相声相关工作日志留存整理



[TOC]



## GPT3-Fine-tune调研

### 基本流程：

1. 准备一个符合格式要求的语料上传。
2. 训练一个新的fine-tune模型。
3. 使用



### 训练价格：

| 模型          | 每1000个token的价格 | 100w个token 4个epoch的开销 |
| ------------- | ------------------- | -------------------------- |
| Davinci[beta] | $0.03               | $120                       |
| Curie         | $0.003              | $12                        |
| Babbage       | $0.0006             | $2.4                       |
| Ada           | $0.0004             | $1.6                       |



当微调一个模型时，每1k个tokens的使用费用大概是推理时使用费用的50%。

100w token大概是5MB的一个文件，4个epoch是他的默认配置



使用fine-tune模型进行推理的价格和使用base模型的架构差不多

推理价格：

| 模型          | 每1000个token的价格 |
| ------------- | ------------------- |
| Davinci[beta] | $0.12               |
| Curie         | $0.012              |
| Babbage       | $0.0024             |
| Ada           | $0.0016             |



**Davinci目前还是beta版本，已经可以使用，但是费率和特性是可能随时变动的。**

Davinci在所有的base模型中的评语是 most powerful。



### 推理参数：

模型在推理时可供调整的参数有：

| 参数              | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| presence_penalty  | -2.0到2.0之间的一个浮点数，如果值为正，则根据最新token是否在之前的文本中出现过来惩罚其本身，可以增加模型谈论新话题的可能性。 |
| frequency_penalty | 在-2.0和2.0之间的数字。如果值为正，则根据到目前为止在文本中出现的频率对新token进行惩罚，从而降低了模型逐字重复同一行的可能性。 |
| temperature       | 使用什么采样温度。数值越高，模型承担的风险越大。更有创意的应用程序可以尝试0.9，答案明确的应用程序可以尝试0。<br />openai建议自己尝试修改这个值或者top_p，但不要同时修改。 |
| top_p             | 一种采样温度的替代方法，称为核抽样，使模型只考虑前top_p的token，0.1等于只考虑概率最高的前10%的token。 |
| n                 | 生成几次补全（慎用，如果设置的多就会根据当前prompt生成多个补全） |



### base模型基本效果对比：

目前对四个base模型进行相声生成的基础效果进行简单对比：

**参数：**

```
temperature=0.5,
max_tokens=384,
top_p=0.1,
frequency_penalty=0.0,
presence_penalty=0.0,
prompt="0:你好!\n1:最近好久不见了，啊。\n0:是啊，好久不见。不过我听说你最近挺忙的。\n1:让你说着了，这个最近啊，还真的非常忙。没办法啊，时代造专家啊。\n0:",
```

效果：

![image-20220226115208642](/Users/wuxiangbo/SAVE_DATA/GPT3/img/image-20220226115208642.png)





**参数：**

```
temperature=0.5,
max_tokens=384,
top_p=0.1,
frequency_penalty=0.5,
presence_penalty=0.0,
prompt="0:你好!\n1:最近好久不见了，啊。\n0:是啊，好久不见。不过我听说你最近挺忙的。\n1:让你说着了，这个最近啊，还真的非常忙。没办法啊，时代造专家啊。\n0:",
```

效果：

![image-20220226113939658](/Users/wuxiangbo/SAVE_DATA/GPT3/img/image-20220226113939658.png)



**参数：**

```
temperature=0.5,
max_tokens=384,
top_p=0.1,
frequency_penalty=0.5,
presence_penalty=0.5,
prompt="0:你好!\n1:最近好久不见了，啊。\n0:是啊，好久不见。不过我听说你最近挺忙的。\n1:让你说着了，这个最近啊，还真的非常忙。没办法啊，时代造专家啊。\n0:",
```

效果：

![image-20220226114513980](/Users/wuxiangbo/SAVE_DATA/GPT3/img/image-20220226114513980.png)







### 准备数据格式

数据必须为jsonl格式，utf-8编码格式，每一行都是一个完整json,示例如下：

```
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```

openai推荐最少使用200个example，每翻一倍数据，效果会有线性提升。

对话格式（自定义对话机器人）的数据输入格式在guideline中是这样建议的：

```
prompts:
	Summary: 历史交互信息的摘要描述
	Specific information: 与对话相关的上下文
	near interaction: 最近的一轮对话
			customer: <message1>  agent:<response1>
			customer: <message2>

completion:
  <response2>
```

示例：

```json
{"prompt":"Summary: <summary of the interaction so far>\n\nSpecific information:<for example order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\nCustomer: <message2>\nAgent:", "completion":" <response2>\n"}
{"prompt":"Summary: <summary of the interaction so far>\n\nSpecific information:<for example order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\nCustomer: <message2>\nAgent: <response2>\nCustomer: <message3>\nAgent:", "completion":" <response3>\n"}
```

准备尝试的相声格式：

```json
{"prompt":"Specific information:一段名称为清水河的对口相声\n\n###\n\n0: 大家好\n1: 给大家拜年了\n0: 这才几月份呐你就拜年\n1:", "completion":" 我这不是紧张嘛\n"}
{"prompt":"Specific information:一段名称为清水河的对口相声\n\n###\n\n0: 大家好\n1: 给大家拜年了\n0: 这才几月份呐你就拜年\n1:我这不是紧张嘛\n0:你紧张什么呀\n1:", "completion":" 好久不上台，生疏\n"}
```

在guideline中有示例的几种格式中（条件生成：标题生成，实体抽取，自定义对话机器人，根据属性生成商品描述），似乎只有上面那种格式符合我们的要求。



### FINETUNE流程：

开始进行finetune：

1.进行数据准备：

```
openai tools fine_tunes.prepare_data -f <数据文件，jsonl格式>
```

这个命令本身接收包含两列的csv,xlsx等格式的文件，但是会将其转化为jsonl，并且会给出一些修改建议。



2.开始进行finetune

```
openai api fine_tunes.create -t <数据文件id或路径> -m <所选择的基模型>
```

这个命令会做三件事情：1.上传数据文件。2.创建finetune任务。3.保持链接直到任务完成。



3.一些在finetune任务发起成功后可能会用到的命令：

```
# 任务异常中断后可重新拉起
openai api fine_tunes.follow -i <finetune任务id>
# 查看所有finetune任务
openai api fine_tunes.list
# 获取任务状态
openai api fine_tunes.get -i <finetune任务id>
# 取消finetune任务
openai api fine_tunes.cancel -i <finetune任务id>

```



4.当确认任务完成后：

```
openai api completions.create -m <FINE_TUNED_MODEL> -p <YOUR_PROMPT>

或python api更换engine字段，即可使用新的模型
```



## 当前实验规划：



### 前提要求：

1.GPT3 base模型推理实验

模型：任一GPT3base模型，之前选用的为Davinci

数据：50篇20句的相声文本，经过人工筛选

所需结果：(1).用于机器指标评估的单句对应生成文本。(2).用于人工评估及敏感词发现的后10句整体生成文本。



2.GPT3 finetune后模型推理实验

模型：任一GPT3base模型

数据：给GPT3的finetune数据，及50篇20句的相声文本

流程：使用选择的GPT3base模型，进行finetune后，使用finetune模型进行推理。

所需结果：(1).用于机器指标评估的单句对应生成文本。(2).用于人工评估及敏感词发现的后10句整体生成文本。





记录：

950条数据 【test_filter_50x20.txt】 ，Davinci

跑一次机器指标计算是 $25.38 左右



