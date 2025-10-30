# coding: UTF-8
import torch
from tqdm import tqdm
import time
import csv
from datetime import timedelta
import os

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    # def read_final_results(file_path):
    #     final_results = {}
    #     with open(file_path, 'r') as file:
    #         for line in file:
    #             parts = line.strip().split('\t')
    #             file_id = int(parts[0])
    #             label = parts[1]
    #             final_results[file_id] = label
    #     return final_results
    #
    # # 测试函数
    # final_results = read_final_results('F:\python\Bert-Chinese-Text-Classification\Bert-Chinese-Text-Classification\datas\mutilModel\\final_results.txt')


    def load_dataset(path, pad_size=32):
        contents = []
        all_contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, content, image_path = row
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                contents.append((token_ids, int(label), seq_len, mask, image_path))
                # print(contents)
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device


    def _to_tensor(self, datas):

        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        #标签
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        image_paths = [_[4] for _ in datas]  # 获取所有图片的路径

        return (x, seq_len, mask, image_paths), y
    '''
    ([101, 2026, 3899, 25999, 2061, 2172, 1045, 2018, 2000, 2689, 11344, 2044, 2023, 2001, 2579, 1012, 1001, 2120, 16168, 10259, 1001, 24184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    0, 
    22, 
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'F:\\python\\Bert-Chinese-Text-Classification\\Bert-Chinese-Text-Classification\\datas\\mutilModel\\data\\8238.jpg')
    '''
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1

            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    '''
    [(1, 'Canvassing with @ahsankhr for @iamIqraKhalid and #RealChange! #cdnpoli'), 
    ([101, 10683, 7741, 2007, 1030, 6289, 8791, 10023, 2099, 2005, 1030, 24264, 4328, 4160, 16555, 8865, 3593, 1998, 1001, 2613, 22305, 2063, 999, 1001, 372
    9, 16275, 10893, 0, 0, 0, 0, 0], 1, 27, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    (0, 'Thx #BC gov for wasting my tax $ to lie to me about #LNG-no #fracking #cdnpoli #BCpoli #
climatechange'), ([101, 16215, 2595, 1001, 4647, 18079, 2005, 18313, 2026, 4171, 1002, 2000, 4682, 2000, 2033, 2055, 1001, 1048, 3070, 1011, 2053, 1001, 25312, 23177, 1001, 3729, 16275, 10893, 1001, 4647, 18155, 2072], 0, 32, [1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])]
    '''
    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, *value, end="\n"):

        current = time.strftime("[%Y-%m-%d %H:%M:%S]")
        s = current
        for v in value:
            s += " " + str(v)
        s += end
        print(s, end="")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s)
