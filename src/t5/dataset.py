'''
Author: anon
Date: 2022-02-09 14:58:23
LastEditors: anon
LastEditTime: 2022-02-09 15:00:21
FilePath: /crosstalk-generation/src/t5/dataset.py
Description: 

Copyright (c) 2022 by anon/Ultrapower, All Rights Reserved. 
'''
from torch.utils.data import Dataset
import torch,json


class MyDataset(Dataset):
    """

    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        json_pairs = json.loads(self.input_list[index])
        
        input_ids = torch.tensor(json_pairs['src'][:self.max_len], dtype=torch.long)
        labels = torch.tensor(json_pairs['tgt'][:self.max_len], dtype=torch.long)
        return input_ids,labels

    def __len__(self):
        return len(self.input_list)
