'''
Author: anon
Date: 2022-02-07 11:17:02
LastEditors: anon
LastEditTime: 2022-02-07 14:31:51
FilePath: /crosstalk-generation/src/t5/t5_tokenizer.py
Description: 

Copyright (c) 2022 by anon/Ultrapower, All Rights Reserved. 
'''

import jieba
from transformers import BertTokenizer

from transformers import MT5ForConditionalGeneration, T5Tokenizer

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens
