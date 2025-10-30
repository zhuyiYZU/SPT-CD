import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader, Dataset
import torch

# 加载预训练的BERT模型和分词器
# model_name = 'bert-base-uncased'
model_name = './model/chinese_bert'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)




# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        text = self.data.iloc[idx, 1]

        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=128,
                                            truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label': label
        }


# 创建数据加载器
# dataset = CustomDataset(data, tokenizer)

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 定义训练过程
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 5
log_every_n_steps = 100
global_step = 0

# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1_Pooling}/{epochs}")
#     for batch in dataloader:
#         optimizer.zero_grad()
#         inputs = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['label']
#
#         outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#
#         if global_step % log_every_n_steps == 0:
#             print(f"Step {global_step}/{len(dataloader) * epochs}, Loss: {loss.item()}")
#
#         global_step += 1_Pooling
#
#     # 保存训练后的模型
#     if epoch + 1_Pooling == epochs :
#         save_path = f"model_epoch_{epoch + 1_Pooling}.pt"
#         torch.save(model.state_dict(), save_path)
#         print(f"Model saved at {save_path}")
#
# print("Training complete.")

# 加载训练后的模型参数
# model.load_state_dict(torch.load('model_epoch_5.pt'))  # 假设模型参数保存在该文件中
model.eval()  # 将模型设置为评估模式

# 词汇表中的词
# vocab_list = tokenizer.get_vocab()
with open('./data/chinese.txt', encoding='utf-8') as f:
    vocab_list = f.readline().split(',')


text = "这是一个关于[MASK]的词，描述的是一些夸张、虚假的新闻标题。"
inputs = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=128, truncation=True)
input_ids = torch.tensor([inputs['input_ids']])
attention_mask = torch.tensor([inputs['attention_mask']])
with torch.no_grad():
    logits = model(input_ids, attention_mask=attention_mask).logits

# 获取 [MASK] 的索引
masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()

# 获取预测的概率分布
predicted_probabilities = torch.softmax(logits[0, masked_index], dim=-1).tolist()



# 将词和概率对应起来并排序
word_prob_pairs = [(word, probability) for word, probability in zip(vocab_list, predicted_probabilities)]
sorted_word_prob_pairs = sorted(word_prob_pairs, key=lambda pair: pair[1], reverse=True)

print("Predicted words sorted by probability:")
for word, probability in sorted_word_prob_pairs:
    print(f"{word}: {probability:.5f}")
