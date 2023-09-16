import torch
from transformers import BertForMaskedLM, BertTokenizer

def predict_words(verbalizer, template):
    # 加载bert模型和tokenizer
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 加载标签词及其扩展词
    # verbalizer = {}
    # with open(verbalizer_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         label_word, extend_words = line.strip().split(':')
    #         extend_words = extend_words.split(',')
    #         verbalizer[label_word] = extend_words


    # print(verbalizer)
    # 对每个标签词及其扩展词，用bert预测其在模板中出现的概率
    result = {}
    for label_word, extend_words in verbalizer.items():
        for word in [label_word] + extend_words:
            masked_template = template.replace('[MASK]', word)
            # print(masked_template)
            input_ids = torch.tensor(tokenizer.encode(masked_template)).unsqueeze(0)
            # print(input_ids)
            outputs = model(input_ids=input_ids)

            # print(outputs)
            prediction_scores = outputs.logits
            # print(prediction_scores)
            masked_index = masked_template.find('[MASK]')
            probabilities = torch.softmax(prediction_scores[0, masked_index], dim=-1)
            result[word] = probabilities[tokenizer.convert_tokens_to_ids(word)].item()

    # 对结果按概率排序并返回
    return sorted(result.items(), key=lambda x: x[1], reverse=True)
dic1 = {
     '标题党': ['题文相符']}
dic2 = {'非标题党': ['题文不符']}
pos_result = predict_words(dic1, '这是一条[MASK]评论，它的内容赞美的。')
neg_result = predict_words(dic2, '这是一条[MASK]评论，它的内容批判的。')

print(pos_result)
print(neg_result)
