# -*- coding: utf-8 -*-
import csv
import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""
'''
[tensor([ 525, 3396, 1874, 1419, 4442, 1317, 1574, 1258, 1861, 3558,   19,   22,
         413,    1, 1926,  166, 2047, 1937,  294,  677]), tensor([14])]
'''

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '点击诱饵', 1: '非点击诱饵'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    # def load(self):
    #     self.data = []
    #     with open(self.path, encoding="utf8") as f:
    #         for line in f:
    #             line = json.loads(line)
    #             tag = line["tag"]
    #             label = self.label_to_index[tag]
    #             title = line["title"]
    #             if self.config["model_type"] == "bert":
    #                 input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
    #             else:
    #                 input_id = self.encode_sentence(title)
    #             input_id = torch.LongTensor(input_id)
    #             label_index = torch.LongTensor([label])
    #             self.data.append([input_id, label_index])
    #         print(self.data)
    #     return
    def load(self):
        self.data = []
        # print(self.path)
        with open(self.path, encoding="utf8") as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                # print(row)
                label, title = row
                input_text = title
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(input_text, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(input_text)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([int(label)])
                self.data.append([input_id, label_index])
            # print(self.data)
        return
    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

# if __name__ == "__main__":
    # from config import Config
    # dg = DataGenerator("../data/wechat_clickbait/train.csv", Config)
    # print(dg[1])
