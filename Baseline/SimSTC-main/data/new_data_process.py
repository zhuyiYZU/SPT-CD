import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
import os
import json
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
import torch
import torch.nn.functional as F  # 用于填充操作

# ==== 初始化 ====
data_file = './sina_data/train.csv'  # 输入的测试数据 CSV 文件路径
save_dir = './sina_data/'
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(data_file, header=None, names=['label', 'title', 'content'])
titles = df['title'].astype(str).values
labels_raw = df['label'].keys()
n_query = len(titles)

# 检查标签的类型
print("标签类型:", type(labels_raw[0]))
# ==== 标签编码 ====
le = LabelEncoder()
labels = le.fit_transform(labels_raw)
# print("编码后的标签:", labels)  # 打印编码后的标签
# print("LabelEncoder 映射关系:", le.classes_)  # 打印每个标签对应的编码

with open(os.path.join(save_dir, 'labels.json'), 'w', encoding='utf-8') as f:
    json.dump(labels.tolist(), f, ensure_ascii=False)

# ==== TF-IDF 向量 ====
tfidf_dim = 300  # 确保 tfidf 特征维度是 428  max_features=tfidf_dim,
vectorizer = TfidfVectorizer(min_df=1,max_features=tfidf_dim)
tfidf_matrix = vectorizer.fit_transform(titles)
vocab = vectorizer.get_feature_names_out()
word2id = {w: i for i, w in enumerate(vocab)}

# 保存 word_emb
word_emb = tfidf_matrix.T.toarray().astype(np.float32)
word_emb_tensor = torch.tensor(word_emb).float()  # 转换为 Tensor

# 确保所有的 word_emb 在特征维度上大小一致
# max_word_dim = word_emb_tensor.size(1)  # 获取最大的维度
# word_emb_padded = F.pad(word_emb_tensor, (0, max_word_dim - word_emb_tensor.size(1)))  # 填充到一致的大小

with open(os.path.join(save_dir, 'word_emb.pkl'), 'wb') as f:
    pkl.dump(word_emb_tensor, f)

# 构建 adj_query2word
q_ids, w_ids = tfidf_matrix.nonzero()
values = tfidf_matrix.data
adj_q2w = coo_matrix((values, (q_ids, w_ids)), shape=tfidf_matrix.shape)
with open(os.path.join(save_dir, 'adj_query2word.pkl'), 'wb') as f:
    pkl.dump(adj_q2w, f)

# 构建 adj_word 共现图
word_word = (tfidf_matrix.T @ tfidf_matrix).toarray()
np.fill_diagonal(word_word, 0)
ww_i, ww_j = np.where(word_word > 0.1)
adj_word = coo_matrix((word_word[ww_i, ww_j], (ww_i, ww_j)), shape=(len(vocab), len(vocab)))
with open(os.path.join(save_dir, 'adj_word.pkl'), 'wb') as f:
    pkl.dump(adj_word, f)

# ==== 构建 adj_query2entity ====
entity2id = {}
q_e_row, q_e_col = [], []
for q_idx, title in enumerate(titles):
    for word, flag in pseg.cut(title):
        if flag in ['nr', 'ns', 'nt']:
            if word not in entity2id:
                entity2id[word] = len(entity2id)
            q_e_row.append(q_idx)
            q_e_col.append(entity2id[word])
adj_q2e = coo_matrix((np.ones(len(q_e_row)), (q_e_row, q_e_col)), shape=(n_query, len(entity2id)))
with open(os.path.join(save_dir, 'adj_query2entity.pkl'), 'wb') as f:
    pkl.dump(adj_q2e, f)

entity_dim = 100
# entity_emb, 这里将维度调整为 428
entity_emb = np.random.randn(len(entity2id), entity_dim).astype(np.float32)
# entity_emb = np.random.randn(len(entity2id)).astype(np.float32)
entity_emb_tensor = torch.tensor(entity_emb).float()  # 转换为 Tensor

# 确保所有的 entity_emb 在特征维度上大小一致
max_entity_dim = entity_emb_tensor.size(1)  # 获取最大的维度
entity_emb_padded = F.pad(entity_emb_tensor, (0, max_entity_dim - entity_emb_tensor.size(1)))  # 填充到一致的大小

with open(os.path.join(save_dir, 'entity_emb.pkl'), 'wb') as f:
    pkl.dump(entity_emb_padded, f)

# ==== 构建 adj_query2tag ====
# 初始化数据
n_query = len(titles)  # 查询数
tag2id = {}
q_t_row, q_t_col, values = [], [], []

# 计算 TF-IDF 值
vectorizer = TfidfVectorizer(max_features=5000)  # 使用 TF-IDF，取前5000个词
tfidf_matrix = vectorizer.fit_transform(titles)

# 获取词汇和标签
for q_idx, title in enumerate(titles):
    tags = jieba.analyse.extract_tags(title, topK=5)
    for tag in tags:
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)  # 为每个标签分配唯一ID
        q_t_row.append(q_idx)  # 查询的索引
        q_t_col.append(tag2id[tag])  # 标签的索引

        # 获取对应的 TF-IDF 值
        # 在 tfidf_matrix 中，行表示查询，列表示词汇，值是该词在该查询中的TF-IDF值
        tag_index = vectorizer.vocabulary_.get(tag)
        if tag_index is not None:
            tfidf_value = tfidf_matrix[q_idx, tag_index]  # 获取该标签在对应查询中的TF-IDF值
            values.append(tfidf_value)

# 创建一个全零的矩阵
adj_q2t_dense = np.zeros((n_query, len(tag2id)), dtype=np.float32)

# 填充非零值（TF-IDF 权重）
for row, col, value in zip(q_t_row, q_t_col, values):
    adj_q2t_dense[row, col] = value


# 保存为 pkl 文件
with open(os.path.join(save_dir, 'adj_query2tag.pkl'), 'wb') as f:
    pkl.dump(adj_q2t_dense, f)


# ==== 构建 adj_tag 共现图 ====
tag_matrix = coo_matrix((np.ones(len(q_t_row)), (q_t_row, q_t_col)), shape=(n_query, len(tag2id)))
tag_tag = (tag_matrix.T @ tag_matrix).toarray()
np.fill_diagonal(tag_tag, 0)
tt_i, tt_j = np.where(tag_tag > 0.1)
adj_tag = coo_matrix((tag_tag[tt_i, tt_j], (tt_i, tt_j)), shape=(len(tag2id), len(tag2id)))
with open(os.path.join(save_dir, 'adj_tag.pkl'), 'wb') as f:
    pkl.dump(adj_tag, f)

# ==== 划分索引 ====
# 生成标签到索引的映射
label_to_idx = {}
for i, lbl in enumerate(labels):
    label_to_idx.setdefault(lbl, []).append(i)
print(label_to_idx)
# 初始化数据索引
train_idx, test_idx = [], []

# 遍历每个标签，按比例划分数据集
for lbl, idxs in label_to_idx.items():
    total_samples = len(idxs)
    print(f"Label: {lbl}, Total samples: {total_samples}")

    np.random.shuffle(idxs)  # 打乱索引
    train_end = int(0.7 * total_samples)  # 70% 的样本作为训练集

    # 按照 70%（训练集）和 30%（测试集）划分
    train_idx += idxs[:train_end]
    test_idx += idxs[train_end:]


# 保存划分后的索引
with open(os.path.join(save_dir, 'train_idx.json'), 'w') as f:
    json.dump(train_idx, f)

with open(os.path.join(save_dir, 'test_idx.json'), 'w') as f:
    json.dump(test_idx, f)

print('✅ 新数据集处理完成，格式已完全对齐 SimSTC')
print('保存目录：', save_dir)
#
#

# import pandas as pd
# import numpy as np
# import jieba
# import jieba.analyse
# import jieba.posseg as pseg
# import os
# import json
# import pickle as pkl
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from scipy.sparse import coo_matrix
# import torch
# import torch.nn.functional as F  # 用于填充操作
#
# # ==== 初始化 ====
# train_file = './tencent_data/train.csv'  # 输入的训练数据 CSV 文件路径
# test_file = './tencent_data/test.csv'  # 输入的测试数据 CSV 文件路径
# save_dir = './tencent_data/'
# os.makedirs(save_dir, exist_ok=True)
#
# # 读取训练集数据
# train_df = pd.read_csv(train_file, header=None, names=['label', 'title', 'content'])
# train_titles = train_df['title'].astype(str).values
# train_labels_raw = train_df['label'].values
# n_train_query = len(train_titles)
#
# # 读取测试集数据
# test_df = pd.read_csv(test_file, header=None, names=['label', 'title', 'content'])
# test_titles = test_df['title'].astype(str).values
# test_labels_raw = test_df['label'].values
# n_test_query = len(test_titles)
#
# # 合并训练集和测试集，用于统一的特征提取
# all_titles = np.concatenate([train_titles, test_titles])
# all_labels_raw = np.concatenate([train_labels_raw, test_labels_raw])
# all_n_query = len(all_titles)
#
# # ==== 标签编码 ====
# le = LabelEncoder()
# all_labels = le.fit_transform(all_labels_raw)
#
# # 保存标签
# with open(os.path.join(save_dir, 'labels.json'), 'w', encoding='utf-8') as f:
#     json.dump(all_labels.tolist(), f, ensure_ascii=False)
#
# # ==== TF-IDF 向量 ====
# tfidf_dim = 300  # 确保 tfidf 特征维度是 428
# vectorizer = TfidfVectorizer(min_df=1, max_features=tfidf_dim)
# tfidf_matrix = vectorizer.fit_transform(all_titles)
# vocab = vectorizer.get_feature_names_out()
# word2id = {w: i for i, w in enumerate(vocab)}
#
# # 保存 word_emb
# word_emb = tfidf_matrix.T.toarray().astype(np.float32)
# word_emb_tensor = torch.tensor(word_emb).float()  # 转换为 Tensor
#
# with open(os.path.join(save_dir, 'word_emb.pkl'), 'wb') as f:
#     pkl.dump(word_emb_tensor, f)
#
# # 构建 adj_query2word
# q_ids, w_ids = tfidf_matrix.nonzero()
# values = tfidf_matrix.data
# adj_q2w = coo_matrix((values, (q_ids, w_ids)), shape=tfidf_matrix.shape)
# with open(os.path.join(save_dir, 'adj_query2word.pkl'), 'wb') as f:
#     pkl.dump(adj_q2w, f)
#
# # 构建 adj_word 共现图
# word_word = (tfidf_matrix.T @ tfidf_matrix).toarray()
# np.fill_diagonal(word_word, 0)
# ww_i, ww_j = np.where(word_word > 0.1)
# adj_word = coo_matrix((word_word[ww_i, ww_j], (ww_i, ww_j)), shape=(len(vocab), len(vocab)))
# with open(os.path.join(save_dir, 'adj_word.pkl'), 'wb') as f:
#     pkl.dump(adj_word, f)
#
# # ==== 构建 adj_query2entity ====
# entity2id = {}
# q_e_row, q_e_col = [], []
# for q_idx, title in enumerate(all_titles):
#     for word, flag in pseg.cut(title):
#         if flag in ['nr', 'ns', 'nt']:
#             if word not in entity2id:
#                 entity2id[word] = len(entity2id)
#             q_e_row.append(q_idx)
#             q_e_col.append(entity2id[word])
# adj_q2e = coo_matrix((np.ones(len(q_e_row)), (q_e_row, q_e_col)), shape=(all_n_query, len(entity2id)))
# with open(os.path.join(save_dir, 'adj_query2entity.pkl'), 'wb') as f:
#     pkl.dump(adj_q2e, f)
#
# entity_dim = 100
# # entity_emb, 这里将维度调整为 `100`
# entity_emb = np.random.randn(len(entity2id), entity_dim).astype(np.float32)
# entity_emb_tensor = torch.tensor(entity_emb).float()  # 转换为 Tensor
#
# with open(os.path.join(save_dir, 'entity_emb.pkl'), 'wb') as f:
#     pkl.dump(entity_emb_tensor, f)
#
# # ==== 构建 adj_query2tag ====
# n_query = all_n_query  # 查询数
# tag2id = {}
# q_t_row, q_t_col, values = [], [], []
#
# # 计算 TF-IDF 值
# vectorizer = TfidfVectorizer(max_features=5000)  # 使用 TF-IDF，取前5000个词
# tfidf_matrix = vectorizer.fit_transform(all_titles)
#
# # 获取词汇和标签
# for q_idx, title in enumerate(all_titles):
#     tags = jieba.analyse.extract_tags(title, topK=5)
#     for tag in tags:
#         if tag not in tag2id:
#             tag2id[tag] = len(tag2id)  # 为每个标签分配唯一ID
#         q_t_row.append(q_idx)  # 查询的索引
#         q_t_col.append(tag2id[tag])  # 标签的索引
#
#         # 获取对应的 TF-IDF 值
#         tag_index = vectorizer.vocabulary_.get(tag)
#         if tag_index is not None:
#             tfidf_value = tfidf_matrix[q_idx, tag_index]  # 获取该标签在对应查询中的TF-IDF值
#             values.append(tfidf_value)
#
# # 创建一个全零的矩阵
# adj_q2t_dense = np.zeros((n_query, len(tag2id)), dtype=np.float32)
#
# # 填充非零值（TF-IDF 权重）
# for row, col, value in zip(q_t_row, q_t_col, values):
#     adj_q2t_dense[row, col] = value
#
# # 保存为 pkl 文件
# with open(os.path.join(save_dir, 'adj_query2tag.pkl'), 'wb') as f:
#     pkl.dump(adj_q2t_dense, f)
#
# # 构建 adj_tag 共现图
# tag_matrix = coo_matrix((np.ones(len(q_t_row)), (q_t_row, q_t_col)), shape=(n_query, len(tag2id)))
# tag_tag = (tag_matrix.T @ tag_matrix).toarray()
# np.fill_diagonal(tag_tag, 0)
# tt_i, tt_j = np.where(tag_tag > 0.1)
# adj_tag = coo_matrix((tag_tag[tt_i, tt_j], (tt_i, tt_j)), shape=(len(tag2id), len(tag2id)))
# with open(os.path.join(save_dir, 'adj_tag.pkl'), 'wb') as f:
#     pkl.dump(adj_tag, f)
#
# # ==== 使用给定的 train 和 test 文件 ====
# train_idx = list(range(n_train_query))  # 假设训练集的索引是前面的样本
# test_idx = list(range(n_train_query, n_train_query + n_test_query))  # 测试集的索引是后面的样本
#
# # 保存划分后的索引
# with open(os.path.join(save_dir, 'train_idx.json'), 'w') as f:
#     json.dump(train_idx, f)
#
# with open(os.path.join(save_dir, 'test_idx.json'), 'w') as f:
#     json.dump(test_idx, f)
#
# print('✅ 新数据集处理完成，格式已完全对齐 SimSTC')
# print('保存目录：', save_dir)
