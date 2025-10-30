import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

'''
无监督对比学习框架，核心部分是通过多视角学习和对比学习来获得文本的有效表示。Classifier负责将输入特征投影到目标空间，而UCL负责计算不同视角之间的对比损失并优化模型。
'''

class Classifier(nn.Module):
    """
    Classifier是一个简单的全连接网络，旨在对输入特征进行投影，得到目标特征空间中的表示。
    它通过nn.Sequential构建了一个包含两层全连接层（Linear），中间加了BatchNorm1d和ReLU激活函数，最后将输入数据投影到输出特征空间。
    输入：doc_fea表示输入的文本特征。
    输出：返回经过normalize处理后的投影特征z，标准化的目的是将特征向量的范数调整为1，从而适应对比学习。
    """

    def __init__(self, in_fea, hid_fea, out_fea, drop_out=0.5):
        # super(Classifier, self).__init__()
        super(Classifier, self).__init__()
        print(f"🌟 Rebuilding Classifier with in_fea={in_fea}")
        self.projector = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, out_fea))

        # print(f"[✅ Classifier] Linear 1 weight shape: {self.projector[0].weight.shape}")
        # print(f"[✅ Classifier] Linear 2 weight shape: {self.projector[3].weight.shape}")

    def forward(self, doc_fea):
        z = F.normalize(self.projector(doc_fea),dim=1)
        return z


# class UCL(nn.Module):
#     def __init__(self, in_fea, out_fea, temperature=0.5):
#         super(UCL, self).__init__()
#         self.projector = nn.Sequential(   #标准的投影网络，将输入特征映射到目标特征空间
#             nn.Linear(in_fea, out_fea),
#             nn.BatchNorm1d(out_fea),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_fea, out_fea))
#         self.projector_2 = nn.Sequential(  #处理的是较为复杂的输入（比如输入特征维度较大时，in_fea + 300）
#             nn.Linear(in_fea, out_fea),
#             nn.BatchNorm1d(out_fea),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_fea, out_fea))
#
#         self.tem = temperature   #temperature（温度参数）在对比学习中用于调节相似度的尺度，通常会影响计算相似度时的平衡。
#         self.hidden_fea = in_fea
#
#     def sim(self, z1: torch.Tensor, z2: torch.Tensor):
#         '''
#         sim方法计算两个张量（z1和z2）之间的余弦相似度，即计算它们的内积。
#         在计算前，先对z1和z2进行标准化。
#         '''
#         z1 = F.normalize(z1)
#         z2 = F.normalize(z2)
#         return torch.mm(z1, z2.t())
#
#     def forward(self, doc_fea):
#         total_loss = 0
#         '''
#         forward方法实现了模型的前向传播，其中涉及到计算对比损失（contrastive loss）。
#         它通过两个嵌套循环选择doc_fea中的不同视角（例如词、词性、实体视角）进行对比学习，分别得到两个视角的投影结果（out_1和out_2），并将其标准化。
#         然后，使用batch_loss函数计算两个视角之间的损失，最后将所有的损失合并，得到最终的total_loss。
#         '''
#         for i in range(2):
#             for j in range(i+1, 3):
#                 print(f"doc_fea[{j}] shape: {doc_fea[j].shape[1]}")
#                 # print(' doc_fea[i].shape[1]', self.hidden_fea)
#                 out_1 = self.projector(doc_fea[i]) if doc_fea[i].shape[1] == self.hidden_fea else self.projector_2(doc_fea[i])
#                 out_2 = self.projector(doc_fea[j]) if doc_fea[j].shape[1] == self.hidden_fea else self.projector_2(doc_fea[j])
#
#                 out_1, out_2 = F.normalize(out_1, dim=1), F.normalize(out_2, dim=1)
#
#                 out = torch.cat([out_1, out_2], dim=0)
#                 dim = out.shape[0]
#
#                 batch_size = 5120 #* 2 #2560
#                 l1 = self.batch_loss(out_1, out_2, batch_size)
#                 l2 = self.batch_loss(out_2, out_1, batch_size)
#
#                 loss = (l1+l2) * 0.5
#                 total_loss += loss.mean()
#                 return total_loss
#
#     def batch_loss(self, out_1, out_2, batch_size):
#         '''
#         batch_loss方法计算了对比学习损失，它的主要目标是计算同一批次中正样本和负样本的相似度损失。
#         具体来说，对于每一批数据，它计算：
#         refl_sim：同一视角下的相似度（即正样本之间的相似度）。
#         between_sim：不同视角之间的相似度（即负样本之间的相似度）。
#         通过这些相似度计算损失，最后将所有批次的损失合并，得到最终的对比损失。
#         '''
#         device = out_1.device
#         num_nodes = out_1.size(0)
#         num_batches = (num_nodes - 1) // batch_size + 1
#         f = lambda x: torch.exp(x / self.tem)
#         indices = torch.arange(0, num_nodes).to(device)
#         losses = []
#
#         for i in range(num_batches):
#             mask = indices[i * batch_size:(i + 1) * batch_size]
#             refl_sim = f(self.sim(out_1[mask], out_1))  # [B, N]
#             between_sim = f(self.sim(out_1[mask], out_2))  # [B, N]
#
#             losses.append(-torch.log(
#                 between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
#                 / (refl_sim.sum(1) + between_sim.sum(1)
#                    - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
#
#         return torch.cat(losses)

class UCL(nn.Module):
    def __init__(self, in_fea, out_fea, temperature=0.5):
        super(UCL, self).__init__()
        self.out_fea = out_fea
        self.tem = temperature  # 温度参数在对比学习中用于调节相似度的尺度
        self.hidden_fea = in_fea

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        '''
        sim方法计算两个张量（z1和z2）之间的余弦相似度，即计算它们的内积。
        在计算前，先对z1和z2进行标准化。
        '''
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # def forward(self, doc_fea):
    #     total_loss = 0
    #     device = doc_fea[0].device  # 获取第一个输入张量的设备
    #     '''
    #     forward方法实现了模型的前向传播，其中涉及到计算对比损失（contrastive loss）。
    #     它通过两个嵌套循环选择doc_fea中的不同视角进行对比学习，分别得到两个视角的投影结果，并将其标准化。
    #     '''
    #     for i in range(2):
    #         for j in range(i + 1, 3):
    #             # print(f"doc_fea[{j}] shape: {doc_fea[j].shape[1]}")
    #
    #             # 动态选择投影器并确保输入张量和模型在同一设备上
    #             out_1 = self._get_projector(doc_fea[i].shape[1]).to(device)(doc_fea[i].to(device))
    #             out_2 = self._get_projector(doc_fea[j].shape[1]).to(device)(doc_fea[j].to(device))
    #
    #             out_1, out_2 = F.normalize(out_1, dim=1), F.normalize(out_2, dim=1)
    #
    #             out = torch.cat([out_1, out_2], dim=0)
    #
    #             batch_size = 64  # 假设批量大小为5120
    #             l1 = self.batch_loss(out_1, out_2, batch_size)
    #             l2 = self.batch_loss(out_2, out_1, batch_size)
    #             batch_loss = (l1 + l2) * 0.5
    #
    #             batch_loss.mean().backward()  # ✅ 推荐
    #
    #             # batch_loss.backward()  # ⬅️ 单独反向传播
    #             total_loss += batch_loss.detach().mean().item()
    #
    #             # total_loss += batch_loss.detach().item()  # ✅ 累加数值，释放计算图
    #
    #             # loss = (l1 + l2) * 0.5
    #             # total_loss += loss.mean()
    #
    #     return total_loss

    def forward(self, doc_fea):
        device = doc_fea[0].device
        total_loss = 0  # 必须是 tensor，而不是纯数值
        count = 0

        for i in range(2):
            for j in range(i + 1, 3):
                out_1 = self._get_projector(doc_fea[i].shape[1]).to(device)(doc_fea[i].to(device))
                out_2 = self._get_projector(doc_fea[j].shape[1]).to(device)(doc_fea[j].to(device))

                out_1, out_2 = F.normalize(out_1, dim=1), F.normalize(out_2, dim=1)

                batch_size = 64
                l1 = self.batch_loss(out_1, out_2, batch_size)
                l2 = self.batch_loss(out_2, out_1, batch_size)

                batch_loss = (l1 + l2) * 0.5
                total_loss = total_loss + batch_loss.mean()  # 保持为tensor，保留图
                count += 1

        return total_loss / count  # 返回可反向传播的 scalar tensor

    def _get_projector(self, in_fea):
        '''
        根据输入特征维度动态生成投影器（projector）。
        '''
        return nn.Sequential(
            nn.Linear(in_fea, self.out_fea),
            nn.BatchNorm1d(self.out_fea),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_fea, self.out_fea)
        )

    # def batch_loss(self, out_1, out_2, batch_size):
    #     '''
    #     batch_loss方法计算对比学习损失，计算同一批次中正样本和负样本的相似度损失。
    #     '''
    #     device = out_1.device
    #     num_nodes = out_1.size(0)
    #     num_batches = (num_nodes - 1) // batch_size + 1
    #     f = lambda x: torch.exp(x / self.tem)
    #     indices = torch.arange(0, num_nodes).to(device)
    #     losses = []
    #
    #     for i in range(num_batches):
    #         mask = indices[i * batch_size:(i + 1) * batch_size]
    #         print("out_1[mask].shape",out_1[mask].shape)
    #
    #         print("out_1.shape", out_1.shape)
    #         print("out_2.shape", out_2.shape)
    #         refl_sim = f(self.sim(out_1[mask], out_1))  # 正样本相似度
    #         between_sim = f(self.sim(out_1[mask], out_2))  # 负样本相似度
    #
    #         losses.append(-torch.log(
    #             between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
    #             / (refl_sim.sum(1) + between_sim.sum(1)
    #                - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
    #
    #     return torch.cat(losses)

    def batch_loss(self, out_1, out_2, batch_size):
        import gc
        import torch

        device = out_1.device
        num_nodes = out_1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tem)
        indices = torch.arange(0, num_nodes, device=device)
        losses = []

        # 💡 打印初始输入大小
        # print(f"[UCL] Input out_1 shape: {out_1.shape}, out_2 shape: {out_2.shape}")
        # print(f"[UCL] batch_size: {batch_size}, num_batches: {num_batches}")

        # ✂️ 限制最大节点数以避免 OOM
        max_compare = 5000
        if out_1.size(0) > max_compare:
            idx = torch.randperm(out_1.size(0), device=device)[:max_compare]
            out_1 = out_1[idx]
            out_2 = out_2[idx]
            num_nodes = out_1.size(0)
            num_batches = (num_nodes - 1) // batch_size + 1
            indices = torch.arange(0, num_nodes, device=device)
            # print(f"[UCL] ⚠️ Truncated to max_nodes={max_compare}, new shape: {out_1.shape}")

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]

            # print(f"[UCL][Batch {i + 1}/{num_batches}] Computing sim for batch size: {mask.shape[0]}")
            # print(f"  → out_1[mask]: {out_1[mask].shape}, vs out_1: {out_1.shape}")

            refl_sim = f(self.sim(out_1[mask], out_1))
            between_sim = f(self.sim(out_1[mask], out_2))

            # print(f"  ✅ refl_sim.shape: {refl_sim.shape}, between_sim.shape: {between_sim.shape}")
            # print(f"  🔍 refl_sim max: {refl_sim.max().item():.4f}, mean: {refl_sim.mean().item():.4f}")
            # print(f"  🔍 between_sim max: {between_sim.max().item():.4f}, mean: {between_sim.mean().item():.4f}")

            loss = -torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())
            )

            # print(f"  📉 batch loss mean: {loss.mean().item():.4f}")
            losses.append(loss)

            # ✅ 主动释放显存
            del refl_sim, between_sim, loss
            torch.cuda.empty_cache()
            gc.collect()

            # 💻 打印当前显存使用情况
            allocated = torch.cuda.memory_allocated(device=device) / 1024 ** 2
            reserved = torch.cuda.memory_reserved(device=device) / 1024 ** 2
            # print(f"  📊 CUDA mem: allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")

        return torch.cat(losses)



