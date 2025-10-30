from transformers import AutoModel, AutoTokenizer
import torch

# 初始化BERT模型和分词器
model_name = "D:\python_project\cut_word\model\chinese_bert"  # 预训练模型
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 输入句子
input_sentence = "这是一个关于[MASK]的词，描述的是一些夸张、虚假的新闻标题。"

# 分词句子
tokens = tokenizer.tokenize(input_sentence)
masked_index = tokens.index("[MASK]")  # 使用[MASK]标记来标识需要替代的词汇

# 存储扩展词汇及其对应的损失
with open('./data/adj_COLD_chinese.txt', encoding='utf-8') as f:
    extension_words = f.readline().split(',')
# extension_words = ["politics", "part", "government", "diplomatic", "law", "aristotle", "diplomatical", "governance"]
extension_losses = {}

def compute_loss(outputs, masked_index, target_ids):


    # 使用pooler_output计算损失
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs['pooler_output'].view(1, -1), target_ids.view(1))

    return loss

# 遍历每个扩展词
for word in extension_words:
    # 复制原始句子并将扩展词放入[MASK]位置
    masked_tokens = tokens.copy()
    masked_tokens[masked_index] = word

    # 将tokens转换为输入IDs
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # 创建输入张量
    input_tensor = torch.tensor([input_ids])

    # 使用BERT模型计算损失
    with torch.no_grad():
        outputs = model(input_tensor)
        target_id = tokenizer.convert_tokens_to_ids(tokens[masked_index])  # 目标词汇的ID
        loss = compute_loss(outputs, masked_index, torch.tensor([target_id]))  # 使用pooler_output计算损失

    # 存储扩展词的损失
    extension_losses[word] = loss.item()

# 根据损失值对扩展词汇进行排序
sorted_extension_words = sorted(extension_losses.items(), key=lambda x: x[1])

# 输出排序后的扩展词汇及其损失
print("扩展词排序:")
for word, loss in sorted_extension_words:
    print(f"{word}: Loss = {loss}")
