
import os,sys

sys.path.append('/data1/anon/crosstalk-generation/src/zhouwenwang/Fengshenbang-LM')



from fengshen import RoFormerModel    
from fengshen import RoFormerConfig
from transformers import BertTokenizer
import torch
import numpy as np

max_length = 32

tokenizer = BertTokenizer.from_pretrained("/data1/anon/crosstalk-generation/pretrain_model/zhouwenwang")
config = RoFormerConfig.from_pretrained("/data1/anon/crosstalk-generation/pretrain_model/zhouwenwang")
model = RoFormerModel.from_pretrained("/data1/anon/crosstalk-generation/pretrain_model/zhouwenwang")

sentence = '清华大学在'

def single_line(sentence):
    for i in range(max_length):
        encode = torch.tensor(
            [[tokenizer.cls_token_id]+tokenizer.encode(sentence, add_special_tokens=False)]).long()
        logits = model(encode)[0]
        logits = torch.nn.functional.linear(
            logits, model.embeddings.word_embeddings.weight)
        logits = torch.nn.functional.softmax(
            logits, dim=-1).cpu().detach().numpy()[0]
        sentence = sentence + \
            tokenizer.decode(int(np.random.choice(logits.shape[1], p=logits[-1])))
        if sentence[-1] == '。':
            break
    return sentence
print(single_line(sentence))

print()