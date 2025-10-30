# 函数：从文件中获取实体概念关系
def get_instance_concept(file):
    ent_concept = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            cpt = line[0]
            ent = line[1]
            ent_concept.setdefault(ent, []).append(cpt)
    return ent_concept

file = './data/data-concept-instance-relations.txt'

ent_concept = get_instance_concept(file)

label_words = ['friendly', 'offensive']

entities_with_concepts = {}  # 用于存储实体和对应概念的字典

# 遍历label_words列表
for entity in label_words:
    if entity in ent_concept:
        entities_with_concepts[entity] = ent_concept[entity]

# 打印实体和对应的概念
for entity, concepts in entities_with_concepts.items():
    print(f"实体 '{entity}' 对应的概念: {', '.join(concepts)}")
