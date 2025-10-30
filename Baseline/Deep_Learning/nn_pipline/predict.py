# -*- coding: utf-8 -*-
import torch
from config import Config
from model import TorchModel
from transformers import BertTokenizer

def predict(config, input_string):
    input_id = [BertTokenizer.from_pretrained(config["pretrain_model_path"]).encode(input_string, max_length=config["max_length"], pad_to_max_length=True)]
    input_id = torch.LongTensor(input_id)
    model = TorchModel(config)
    model.load_state_dict(torch.load(config['model_path']))  #加载模型

    model.eval()
    with torch.no_grad():
        # 标识是否使用gpu
        cuda_flag = torch.cuda.is_available()
        if cuda_flag:
            input_id = input_id.cuda()
            model = model.cuda()
        print(input_id.shape)
        y_pred = model(input_id) #预测
        print(y_pred)
        pred_label = torch.argmax(y_pred)
        #tensor([-0.2994, -0.6063], device='cuda:0')
        # values, result = torch.max(y_pred.data, 1)  #返回概率和对应的索引
        print("输入：%s, 结果：%f" % (input_string, pred_label))
    # for i, input_string in enumerate(input_string):
    #
    #     # print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), values[i]) )

if __name__ == "__main__":
    # main(Config)
    title = '中医防癌，从守护肾精、疏肝健脾做起 | 医说新语'
    predict(Config, title)