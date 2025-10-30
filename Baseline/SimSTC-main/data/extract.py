import pandas as pd
import os

# 函数：从多个CSV文件中按类别抽取数据，并去重后保存
def extract_and_merge_by_category(files, num_samples_group1, num_samples_group2, output_file_group1, output_file_group2):
    # 创建空的DataFrame来存储所有抽取的数据
    combined_data_group1 = pd.DataFrame(columns=['label', 'title', 'content', 'source'])
    combined_data_group2 = pd.DataFrame(columns=['label', 'title', 'content', 'source'])

    # 存储已抽取的索引，避免重复
    all_extracted_indexes_group1 = set()
    all_extracted_indexes_group2 = set()

    # 总数据量每个文件应该抽取的数量
    num_samples_per_file_group1 = num_samples_group1 // len(files)  # 每个文件抽取125条数据
    num_samples_per_file_group2 = num_samples_group2 // len(files)  # 每个文件抽取250条数据

    for file in files:
        if os.path.exists(file):
            # 读取CSV文件
            data = pd.read_csv(file, header=None, names=['label', 'title', 'content', 'source'])

            # 获取文件中所有类别
            categories = data['label'].unique()

            # 针对每个文件，先抽取500条数据，再抽取1000条数据
            file_data_group1 = pd.DataFrame(columns=['label', 'title', 'content', 'source'])
            file_data_group2 = pd.DataFrame(columns=['label', 'title', 'content', 'source'])

            # 按类别计算比例
            category_counts = data['label'].value_counts(normalize=True)  # 获取类别的比例

            # 抽取500条数据（group1），确保每个类别的比例一致
            for category, category_ratio in category_counts.items():
                # 按类别筛选数据
                category_data = data[data['label'] == category]
                category_data = category_data[~category_data.index.isin(all_extracted_indexes_group1)]  # 去除已抽取的数据

                # 计算该类别应抽取的数量
                samples_for_category_group1 = int(category_ratio * num_samples_per_file_group1)
                if len(category_data) >= samples_for_category_group1:
                    sampled_category_data_group1 = category_data.sample(n=samples_for_category_group1, random_state=42)
                    file_data_group1 = pd.concat([file_data_group1, sampled_category_data_group1], ignore_index=True)
                    all_extracted_indexes_group1.update(sampled_category_data_group1.index)

            # 将500条数据合并到最终的combined_data_group1中
            combined_data_group1 = pd.concat([combined_data_group1, file_data_group1], ignore_index=True)

            # 抽取1000条数据（group2），确保没有与第一组重叠
            for category, category_ratio in category_counts.items():
                # 按类别筛选数据
                category_data = data[data['label'] == category]
                category_data = category_data[~category_data.index.isin(all_extracted_indexes_group2)]  # 去除已抽取的数据

                # 计算该类别应抽取的数量
                samples_for_category_group2 = int(category_ratio * num_samples_per_file_group2)
                if len(category_data) >= samples_for_category_group2:
                    sampled_category_data_group2 = category_data.sample(n=samples_for_category_group2, random_state=42)
                    file_data_group2 = pd.concat([file_data_group2, sampled_category_data_group2], ignore_index=True)
                    all_extracted_indexes_group2.update(sampled_category_data_group2.index)

            # 将1000条数据合并到最终的combined_data_group2中
            combined_data_group2 = pd.concat([combined_data_group2, file_data_group2], ignore_index=True)

        else:
            print(f"文件 {file} 不存在!")

    # 去重：根据所有列进行去重
    combined_data_group1 = combined_data_group1.drop_duplicates()
    combined_data_group2 = combined_data_group2.drop_duplicates()

    # 保存合并并去重后的数据到两个新的CSV文件
    combined_data_group1.to_csv(output_file_group1, index=False, quoting=1)
    combined_data_group2.to_csv(output_file_group2, index=False, quoting=1)

    print(f"数据已保存到 {output_file_group1} 和 {output_file_group2}")

# 示例使用
# 假设有四个CSV文件: 'file1.csv', 'file2.csv', 'file3.csv', 'file4.csv'
files = ['./sina_data/train.csv', './wechat_data/train.csv', './tencent_data/train.csv', './paper_data/test.csv']
num_samples_group1 = 500  # 第一组数据总数：500
num_samples_group2 = 1000  # 第二组数据总数：1000
output_file_group1 = 'combined_data_group1.csv'  # 第一组数据输出文件名
output_file_group2 = 'combined_data_group2.csv'  # 第二组数据输出文件名

extract_and_merge_by_category(files, num_samples_group1, num_samples_group2, output_file_group1, output_file_group2)
