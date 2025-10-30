import pickle
"""
# 加载 pkl 文件
with open('sina_data/word_emb.pkl', 'rb') as f:
    adj_query2entity = pickle.load(f)

# 输出内容的类型和部分内容（如果是大数据，可以打印部分内容）
print(type(adj_query2entity))
print(adj_query2entity.shape)
"""



import pandas as pd
# from numpy.distutils.conv_template import header

# 读取CSV文件
df = pd.read_csv('./sina_data/test.csv', header=None, names=['label', 'title', 'content', 'source'])

# 去除完全重复的行，只保留第一行
df_cleaned = df.drop_duplicates(subset=['title', 'content'])

# 检查清理后的数据
print(df_cleaned.head())

# 保存清理后的数据
df_cleaned.to_csv('./sina_data/test.csv', index=False,quoting=1,header=None)

