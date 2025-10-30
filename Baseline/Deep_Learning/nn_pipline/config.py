# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/Sina/trainsina200.csv",
    "valid_data_path": "../data/Sina/testsina1.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 3e-3,
    "pretrain_model_path":r"../model",
    "seed": 987,
    'class_num': 2,
    'vocab_size': 4622
}