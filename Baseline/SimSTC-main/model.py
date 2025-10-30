import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import aggregate
from GCN import GCN
'''
过多个图卷积层对图节点进行嵌入学习，并通过聚合操作整合来自不同节点类型的信息。
'''
# class SHINE(nn.Module):
#
#     '''
#
#     '''
#     def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, params):
#         super(SHINE, self).__init__()
#         self.adj = adj_dict  #邻接矩阵字典，用于存储图中每种类型的邻接矩阵。
#         self.feature = features_dict  #特征字典，包含每个节点类型的特征。
#         self.in_features_dim = in_features_dim  #分别表示输入特征维度和输出特征维度的字典。
#         self.out_features_dim = out_features_dim
#         self.type_num = len(params.type_num_node)  #模型的配置参数（例如，节点类型数量、dropout率等）。
#         self.drop_out = params.drop_out
#         self.concat_word_emb = params.concat_word_emb
#
#         self.device = params.device
#         self.GCNs = nn.ModuleList()  #存储一个 GCN（图卷积网络）层
#         self.GCNs_2 = nn.ModuleList()
#
#         for i in range(1, self.type_num):  #根据 type_num（节点类型数）创建多个GCN层，用于处理不同类型的图节点。
#             self.GCNs.append(GCN(self.in_features_dim[i], self.out_features_dim[i]).to(self.device))
#             self.GCNs_2.append(GCN(self.out_features_dim[i], self.out_features_dim[i]).to(self.device))
#
#     def embed_component(self, norm=True):  #生成每种类型节点的嵌入
#         output = []
#         for i in range(self.type_num - 1):
#             # Is this better use to concat the identity and word embeddings before aggregating?
#             if i == 1 and self.concat_word_emb:
#                 temp_emb = torch.cat([
#                     F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
#                                 self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True)),
#                                 p=self.drop_out, training=self.training), self.feature['word_emb']], dim=-1)
#
#                 output.append(temp_emb)
#             elif i == 0:
#                 temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
#                             self.GCNs[i](self.adj[str(i + 1) + str(i + 1)],self.feature[str(i + 1)], identity=True)),
#                             p=self.drop_out, training=self.training)
#                 output.append(temp_emb)
#             else:
#                 temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
#                                 self.GCNs[i](self.adj[str(i + 1) + str(i + 1)],self.feature[str(i + 1)])),
#                                 p=self.drop_out, training=self.training)
#                 output.append(temp_emb)
#         refined_text_input = aggregate(self.adj, output, self.type_num - 1)   #所有类型的嵌入会通过 aggregate 函数进行聚合，得到最终的文本表示。
#         if norm:
#             refined_text_input_normed = []
#             for i in range(self.type_num - 1):
#                 refined_text_input_normed.append(refined_text_input[i] / (refined_text_input[i].norm(p=2, dim=-1,keepdim=True) + 1e-9))
#         else:
#             refined_text_input_normed = refined_text_input
#         return refined_text_input_normed
#
#     def forward(self, epoch):
#         refined_text_input_normed = self.embed_component()
#
#         return refined_text_input_normed #Doc_features


class SHINE(nn.Module):
    def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, params):
        super(SHINE, self).__init__()
        self.adj = adj_dict  # 邻接矩阵字典
        self.feature = features_dict  # 特征字典
        self.in_features_dim = in_features_dim  # 输入特征维度
        self.out_features_dim = out_features_dim  # 输出特征维度
        self.type_num = len(params.type_num_node)  # 节点类型数量
        self.drop_out = params.drop_out
        self.concat_word_emb = params.concat_word_emb

        self.device = params.device
        self.GCNs = nn.ModuleList()
        self.GCNs_2 = nn.ModuleList()

        # 添加 GCN 层
        for i in range(1, self.type_num):
            self.GCNs.append(GCN(self.in_features_dim[i], self.out_features_dim[i]).to(self.device))
            self.GCNs_2.append(GCN(self.out_features_dim[i], self.out_features_dim[i]).to(self.device))

        # # 动态设置线性层的输入维度（根据拼接后的特征维度）
        # self.linear_projection = nn.Linear(sum(self.out_features_dim), 428)

    def embed_component(self, norm=True):
        output = []
        for i in range(self.type_num - 1):
            if i == 1 and self.concat_word_emb:
                temp_emb = torch.cat([
                    F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True)),
                                p=self.drop_out, training=self.training), self.feature['word_emb']], dim=-1)
                output.append(temp_emb)
            elif i == 0:
                temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                            self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True)),
                            p=self.drop_out, training=self.training)
                output.append(temp_emb)
            else:
                temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)])),
                                p=self.drop_out, training=self.training)
                output.append(temp_emb)

        refined_text_input = aggregate(self.adj, output, self.type_num - 1)

        if norm:
            refined_text_input_normed = []
            for i in range(self.type_num - 1):
                refined_text_input_normed.append(refined_text_input[i] / (refined_text_input[i].norm(p=2, dim=-1, keepdim=True) + 1e-9))
        else:
            refined_text_input_normed = refined_text_input
        # print(f"refined_text_input_normed shape: {refined_text_input_normed.shape}")

        # 将嵌入层的输出通过线性层调整为 428 维
        # final_embedding = self.linear_projection(torch.cat(refined_text_input_normed, dim=-1))

        return refined_text_input_normed

    def forward(self, epoch):
        # 调用 embed_component 获取嵌入并调整维度
        refined_text_input_normed = self.embed_component()

        return refined_text_input_normed  # 返回调整后的嵌入

