# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.models as models

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/mutilModel/train_set.csv'  # 训练集
        self.dev_path = dataset + '/mutilModel/val_set.csv'  # 验证集
        self.test_path = dataset + '/mutilModel/test_set.csv'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/mutilModel/classes.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1


# 实例化 ResNet-50 模型
resnet50 = models.resnet50(pretrained=True)

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

        # 图像
        self.fc_image = nn.Linear(1000, config.num_classes)  # 用于图片特征的全连接层
        self.fc_concat = nn.Linear(4, config.num_classes)  # 用于拼接后的全连接层
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图片大小调整为预训练模型的输入大小
            transforms.ToTensor(),  # 将图片转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图片进行标准化
        ])
        self.image_features_extractor = resnet50  # 使用预训练的ResNet50模型提取图片特征

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled_text = self.bert(context, attention_mask=mask, token_type_ids=None, return_dict=False)
        # encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        #print(pooled_text.shape)  #torch.Size([128, 768])
        out = pooled_text.unsqueeze(1)
        # print(out.shape)  #torch.Size([128, 1, 768])

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        text_output = self.fc_cnn(out)
        print(text_output.shape)

        image_paths = x[3]
        # # 处理图片特征
        image_features = []
        for image_path in image_paths:
            # print(image_path)
            image = Image.open(image_path)  # 打开图片
            image = self.image_transform(image)  # 对图片进行预处理
            image = image.unsqueeze(0)  # 添加一个维度，变成4D张量，适应ResNet的输入格式
            with torch.no_grad():  # 不计算梯度
                image_feature = self.image_features_extractor(image.cuda())  # 提取图片特征

            # print(image_feature.shape)   #torch.Size([1, 1000])

            image_features.append(image_feature)
        image_features = torch.cat(image_features, dim=0)  # 将图片特征拼接成一个张量

        # 图片特征经过全连接层
        image_output = self.fc_image(image_features)
        # print(image_output.shape) #torch.Size([128, 2])

        #
        # 将文本特征和图片特征拼接
        combined_features = torch.cat([text_output, image_output], dim=1)
        # 拼接后的特征经过全连接层
        output = self.fc_concat(combined_features)


        return output
