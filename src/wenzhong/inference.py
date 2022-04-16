from transformers import GPT2Tokenizer, GPT2Model
from transformers import pipeline, set_seed
# tokenizer = GPT2Tokenizer.from_pretrained('/data1/anon/crosstalk-generation/pretrain_model/wenzhong')
# model = GPT2Model.from_pretrained('/data1/anon/crosstalk-generation/pretrain_model/wenzhong')
text = "给大家拜个年."
generator = pipeline('text-generation', model='/data1/anon/crosstalk-generation/pretrain_model/wenzhong')
out = generator("北京位于", max_length=30, num_return_sequences=1)
print(out)