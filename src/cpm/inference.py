'''
Author: anon
Date: 2022-02-07 20:18:18
LastEditors: anon
LastEditTime: 2022-02-10 16:09:11
FilePath: /crosstalk-generation/src/cpm/inference.py
Description: 

Copyright (c) 2022 by anon/Ultrapower, All Rights Reserved. 
'''


from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelWithLMHead,ConversationalPipeline,Conversation

tokenizer = AutoTokenizer.from_pretrained("/data1/anon/crosstalk-generation/pretrain_model/CPM-large")
model = AutoModelWithLMHead.from_pretrained("/data1/anon/crosstalk-generation/pretrain_model/CPM-large")

text_generator = TextGenerationPipeline(model, tokenizer)
text = text_generator('我是吴相博', max_length=50, do_sample=True, top_p=0.9)
print(text)

# conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)
# conversation_1 = Conversation("<cls>今天我俩来给大家说段相声<sep>")
# print()