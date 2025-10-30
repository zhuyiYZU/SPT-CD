# -*- coding: utf-8 -*-
import torch
from loader import load_data
from sklearn.metrics import f1_score, precision_recall_fscore_support
"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc,f1 = self.show_stats()
        print(acc, f1)
        return acc, f1

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    # def show_stats(self):
    #     correct = self.stats_dict["correct"]
    #     wrong = self.stats_dict["wrong"]
    #     self.logger.info("预测集合条目总量：%d" % (correct +wrong))
    #     self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
    #     self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
    #     self.logger.info("--------------------")
    #     return correct / (correct + wrong)

    def show_stats(self):
        true_labels = []
        pred_labels = []
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)
            for true_label, pred_label in zip(labels, pred_results):
                pred_label = torch.argmax(pred_label)
                true_labels.append(int(true_label))
                pred_labels.append(int(pred_label))
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        accuracy = correct / (correct + wrong)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % accuracy)
        self.logger.info("F1分数: %f" % f1)
        self.logger.info("--------------------")
        return accuracy, f1