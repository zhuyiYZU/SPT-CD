import numpy as np
import pickle as pkl
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import pickle as pkl
from scipy import sparse
from sklearn import metrics
from utils import fetch_to_tensor, fetch_to_sparse_tensor
from model import SHINE
from CL import Classifier, UCL

class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.dataset_name = params.dataset
        self.max_epoch = params.max_epoch
        self.save_path = params.save_path
        self.device = params.device
        self.hidden_size = params.hidden_size
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.concat_word_emb = params.concat_word_emb
        self.type_names = params.type_num_node
        self.data_path = params.data_path
        self.dropout = params.drop_out
        
        self.ucl_temp = params.ucl_temp
        # input_dim = self.features_dict['word_emb'].shape[0]  # 例如 30357
        # print(f"[⚙️] reduce_refine1 Input Dim: {input_dim}")
        # self.reduce_refine1 = torch.nn.Linear(input_dim, 128).to(self.device)


        self.alpha = params.alpha

        self.theta = params.theta

        self.adj_dict, self.features_dict, self.train_idx, self.valid_idx, self.test_idx, self.labels, self.nums_node = self.load_data()
        # === 现在 features_dict 已定义，才可以用它 ===
        # input_dim = self.features_dict['word_emb'].shape[0]
        # print(f"[⚙️] reduce_refine1 Input Dim: {input_dim}")
        # self.reduce_refine1 = torch.nn.Linear(input_dim, 128).to(self.device)

        self.reduce_refine1 = None  # 暂不初始化
        self.label_num = len(set(self.labels))
        self.labels = torch.tensor(self.labels).to(self.device)
        self.out_features_dim = [self.label_num, self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size]
        in_fea_final = self.out_features_dim[1] + self.out_features_dim[2] + self.out_features_dim[3]
        self.in_features_dim = [0, self.nums_node[1], self.nums_node[2], self.nums_node[-1], in_fea_final]



        # if self.concat_word_emb:
        #     self.in_features_dim[-1] += self.features_dict['word_emb'].shape[-1]
        # print(f"1\[DEBUG] doc_fea dim (in_features_dim[-1]): {self.in_features_dim[-1]}")
        # print(f"[DEBUG before concat] self.in_features_dim[-1] dim: {self.in_features_dim[-1]}")
        # if self.concat_word_emb:
        #     word_emb_dim = self.features_dict['word_emb'].shape[-1]
        #     self.in_features_dim[-1] += word_emb_dim
        #
        #     print(f"[DEBUG] word_emb dim to concat: {word_emb_dim}")
        # print(f"[DEBUG after concat] final doc_fea dim: {self.in_features_dim[-1]}")
        #
        # print(
        #     f"[DEBUG] init Classifier: in={self.in_features_dim[-1]}, hid={self.out_features_dim[-1]}, out={self.out_features_dim[0]}")

        self.model = SHINE(self.adj_dict, self.features_dict, self.in_features_dim, self.out_features_dim, params)
        self.model = self.model.to(self.device)

        self.cls = Classifier(self.in_features_dim[-1], self.out_features_dim[-1], self.out_features_dim[0], self.dropout)



        self.cls.apply(self._weights_init)  # 强制重置 Linear 权重

        # self.cls = Classifier(doc_fea_dim, self.out_features_dim[-1], self.out_features_dim[0], self.dropout)
        self.cls = self.cls.to(self.device)
        self.ucl = UCL(self.hidden_size, self.hidden_size, self.ucl_temp)
        self.ucl = self.ucl.to(self.device)

        
        self.optim = optim.Adam([{'params': self.model.parameters()},
                         {'params': self.cls.parameters()},
                         {'params': self.ucl.parameters()},
                        ], lr=self.lr, weight_decay=self.weight_decay)

    def _weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    def train(self):
        global_best_acc = 0
        global_best_f1 = 0
        global_best_epoch = 0
        best_test_acc = 0
        best_test_f1 = 0
        best_valid_epoch = 0
        best_valid_f1 = 0
        best_valid_acc = 0
        acc_valid = 0
        loss_valid = 0
        f1_valid = 0
        acc_test = 0
        loss_test = 0
        f1_test = 0
        best_acc = 0
        best_f1 = 0
        for i in range(1, self.max_epoch + 1):
            torch.cuda.empty_cache()

            if self.params.whether_warmup:
                self.adjust_learning_rate(self.optim, i, self.params)
            
            t=time.time()
            refine_doc = self.model(i) # [pos_num x dim, word_num x dim, entity_num x dim]

            if self.reduce_refine1 is None:
                in_dim = refine_doc[1].shape[1]
                # print(f"🔧 Auto init reduce_refine1 with input dim: {in_dim}")
                self.reduce_refine1 = torch.nn.Linear(in_dim, 128).to(self.device)

            refine_doc[1] = self.reduce_refine1(refine_doc[1])
            # for i, x in enumerate(refine_doc):
            #     print(
            #         f"refine_doc[{i}]: shape={x.shape}, device={x.device}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

            doc_fea = torch.cat(refine_doc, dim=-1)

            """
                    if self.concat_word_emb:
                        word_emb_dim = self.features_dict['word_emb'].shape[-1]
                        self.in_features_dim[-1] += word_emb_dim
            """


            # word_emb_dim = self.features_dict['word_emb'].shape[-1]
            #
            # print("word_emb_dim:",word_emb_dim)
            # print("befor doc_fea:", doc_fea.shape)
            # doc_fea += word_emb_dim
            #
            # print("after doc_fea:", doc_fea.shape)
            # print("doc_fea.shape", doc_fea.shape)
            # if self.concat_word_emb:
            #     word_emb = self.features_dict['word_emb'] # shape: [N, 128]
            #     print("word_emb.shape", word_emb.shape)
            #     doc_fea = torch.cat([doc_fea, word_emb], dim=-1)  # shape: [N, 512]
            # print("doc_fea.shape", doc_fea.shape)
            # if self.concat_word_emb:
            #     word_emb = self.features_dict['word_emb']  # [N, 128]
            #     doc_fea = torch.cat([doc_fea, word_emb], dim=-1)  # [N, 512]
            # doc_fea = torch.cat(refine_doc, dim=-1)

            output = self.cls(doc_fea)
            
            train_scores = output[self.train_idx]
            
            train_labels = self.labels[self.train_idx]

            cls_loss = F.cross_entropy(train_scores, train_labels) * self.theta

            # print("refine_doc:", refine_doc[0].shape)
            # print("refine_doc:", refine_doc[1].shape)
            # print("refine_doc:", refine_doc[2].shape)
            # refine_doc: torch.Size([496, 128])
            # refine_doc: torch.Size([496, 624])
            # refine_doc: torch.Size([496, 128])

            ucl_loss = self.ucl(refine_doc)
            loss = cls_loss + self.alpha * ucl_loss
            # 💡 仅对 supervised cross-entropy 做一次 backward
            # loss = cls_loss  # + 0 也可

            self.optim.zero_grad()

            loss.backward()
            self.optim.step()
            loss = loss.item()

            acc = torch.eq(torch.argmax(train_scores, dim=-1), train_labels).float().mean().item()
            # print(f"[Epoch {i}] cls_loss={cls_loss.item():.4f} | ucl_loss={ucl_loss:.4f}")

            print('Epoch {}  train_loss: {:.4f} train_acc: {:.4f} time{:.4f}'.format(i, loss, acc, time.time()-t))
            if i%5 == 0:
                acc_valid, loss_valid, f1_valid, acc_test, loss_test, f1_test = self.test(i)

                if acc_test > global_best_acc:
                    global_best_acc = acc_test
                    global_best_f1 = f1_test
                    global_best_epoch = i
                if acc_valid > best_valid_acc:
                    best_valid_acc = acc_valid
                    best_valid_f1 = f1_valid
                    best_test_acc = acc_test
                    best_test_f1 = f1_test
                    best_valid_epoch = i
                best_acc = global_best_acc
                best_f1 = global_best_f1
                best_epoch = global_best_epoch

                best_acc = global_best_acc
                best_f1 = global_best_f1
                best_epoch = global_best_epoch
                
            if i%100==0:
                print('VALID: VALID ACC', best_valid_acc, ' VALID F1', best_valid_f1, 'EPOCH', best_valid_epoch)
                print('VALID: TEST ACC', best_test_acc, 'TEST F1', best_test_f1, 'EPOCH', best_valid_epoch)
                print('GLOBAL: TEST ACC', global_best_acc, 'TEST F1', global_best_f1, 'EPOCH', global_best_epoch)
        return best_acc, best_f1

    def test(self, epoch):
        t = time.time()
        self.model.training = False
        self.cls.training = False
        self.ucl.training = False
        refine_doc = self.model(epoch)


        if self.reduce_refine1 is None:
            in_dim = refine_doc[1].shape[1]
            # print(f"🔧 Auto init reduce_refine1 with input dim: {in_dim}")
            self.reduce_refine1 = torch.nn.Linear(in_dim, 128).to(self.device)

        refine_doc[1] = self.reduce_refine1(refine_doc[1])


        doc_fea = torch.cat(refine_doc, dim=-1)
        # if self.concat_word_emb:
        #     word_emb = self.features_dict['word_emb'].shape[-1]
        #     doc_fea = torch.cat([doc_fea, word_emb], dim=-1)

        output = self.cls(doc_fea)
        with torch.no_grad():
            valid_scores = output[self.valid_idx]
            valid_labels = self.labels[self.valid_idx]
            loss_valid = F.cross_entropy(valid_scores, valid_labels).item()
            acc_valid = torch.eq(torch.argmax(valid_scores, dim=-1), valid_labels).float().mean().item()
            f1_valid = metrics.f1_score(valid_labels.detach().cpu().numpy(),torch.argmax(valid_scores,-1).detach().cpu().numpy(),average='macro')
            print('Valid  loss: {:.4f}  acc: {:.4f}  f1: {:.4f}'.format(loss_valid, acc_valid, f1_valid))

            test_scores = output[self.test_idx]
            test_labels = self.labels[self.test_idx]
            loss_test = F.cross_entropy(test_scores, test_labels).item()
            acc_test = torch.eq(torch.argmax(test_scores, dim=-1), test_labels).float().mean().item()
            f1_test = metrics.f1_score(test_labels.detach().cpu().numpy(),torch.argmax(test_scores,-1).detach().cpu().numpy(),average='macro')
            print('Test  loss: {:.4f} acc: {:.4f} f1: {:.4f} time: {:.4f}'.format(loss_test, acc_test, f1_test, time.time() - t))
        self.model.training = True
        self.cls.training = True
        self.ucl.training = True
        return acc_valid, loss_valid, f1_valid, acc_test, loss_test, f1_test

    def load_data(self):
        start = time.time()
        adj_dict = {}
        feature_dict = {}
        nums_node = []
        for i in range(1, len(self.type_names)):
            adj_dict[str(0) + str(i)] = pkl.load(
                    open(self.data_path + './adj_{}2{}.pkl'.format(self.type_names[0], self.type_names[i]), 'rb'))
            if i == 1:
                nums_node.append(adj_dict[str(0) + str(i)].shape[0])
            if i != 3:
                adj_dict[str(i) + str(i)] = pkl.load(
                    open(self.data_path + './adj_{}.pkl'.format(self.type_names[i]), 'rb'))
                nums_node.append(adj_dict[str(i) + str(i)].shape[0])
            if i == 3:
                feature_dict[str(i)] = pkl.load(
                    open(self.data_path + './{}_emb.pkl'.format(self.type_names[i]), 'rb'))
                nums_node.append(feature_dict[str(i)].shape[0])
                nums_node.append(feature_dict[str(i)].shape[1])
            else:
                feature_dict[str(i)] = sparse.coo_matrix(np.eye(nums_node[i], dtype=np.float32))

        feature_dict['word_emb'] = torch.tensor(pkl.load(
            open(self.data_path + './word_emb.pkl', 'rb')), dtype=torch.float).to(self.device)

        # print(feature_dict)
        ent_emb=feature_dict['3']
        ent_emb_normed = ent_emb / np.sqrt(np.square(ent_emb).sum(-1, keepdims=True))
        adj_dict['01'] = sparse.coo_matrix(adj_dict['01'], dtype=np.float32)
        adj_dict['11'] = sparse.coo_matrix(adj_dict['11'], dtype=np.float32)

        # print(f"ent_emb_normed type: {type(ent_emb_normed)}")
        # print(f"ent_emb_normed shape: {ent_emb_normed.shape}")

        adj_dict['33'] = np.matmul(ent_emb_normed, ent_emb_normed.T)
        adj_dict['33'] = adj_dict['33'] * np.float32(adj_dict['33'] > 0)
        adj_dict['33'] = sparse.coo_matrix(adj_dict['33'], dtype=np.float32)
        adj_dict['22'] = adj_dict['22'].astype(np.float32)
        adj_dict['02'] = adj_dict['02'].astype(np.float32)
        adj_dict['03'] = adj_dict['03'].astype(np.float32)
        
        adj = {}
        feature = {}
        for i in adj_dict.keys():
            adj[i] = fetch_to_sparse_tensor(adj_dict, i, self.device)
        for i in feature_dict.keys():
            if i == "3" or i == "word_emb":
                feature[i] = fetch_to_tensor(feature_dict, i, self.device)
            else:
                feature[i] = fetch_to_sparse_tensor(feature_dict, i, self.device)
        
        train_set = json.load(open(self.data_path + './train_idx.json'))
        test_set = json.load(open(self.data_path + './test_idx.json'))
        labels = json.load(open(self.data_path + './labels.json'))
        
        data_index = train_set + test_set
        Sumofquery = len(data_index)
        print("data_index_len:",Sumofquery)
        label_dict = {}
        for i in set(labels):
            label_dict[i] = []
        for j, label in enumerate(labels):
            label_dict[label].append(j)
        len_train_idx = len(label_dict) * 20 
        train_list = []
        valid_list = []
        byclass = True if self.dataset_name=='ohsumed' else False 
        if not byclass:
            for i in label_dict.items():
                train_list.append(20)
        else:
            ratio = len_train_idx / Sumofquery
            print("train_ratio: ", ratio)
            residue = []
            max_supp = len_train_idx
            for val in label_dict.values():
                num = math.modf(len(val) * ratio)
                residue.append(num[0])
                train_list.append(int(num[1]))
                max_supp -= int(num[1])
            sorted_list = sorted(range(len(residue)), key= lambda i : residue[i], reverse=True)[ : max_supp]
            for i, val in enumerate(train_list):
                if i in sorted_list:
                    train_list[i] += 1
        valid_list = train_list
        train_idx = []
        valid_idx = []
        test_idx = []
        for i, idxs in enumerate(label_dict.values()):
            np.random.shuffle(idxs)
            for j, idx in enumerate(idxs):
                if j < train_list[i]:
                    train_idx.append(idx)
                elif j >= train_list[i] and j < train_list[i] + valid_list[i]:
                    valid_idx.append(idx)
                else:
                    test_idx.append(idx)
        # print("len(train_idx):",len(train_idx))
        # print("len(valid_idx):",len(valid_idx))
        # print("len(test_idx):",len(test_idx))
        print('data process time: {}'.format(time.time()-start))
        # print(adj[str(0) + str(i + 1)])
        return adj, feature, train_idx, valid_idx, test_idx, labels, nums_node

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.model.state_dict(), path + './model/'.format(self.dataset_name))

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))

    def adjust_learning_rate(self, optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        lr = args.lr
        if epoch < args.warmup_epochs:
            lr = args.lr / args.warmup_epochs * (epoch + 1)
        elif args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.max_epoch - args.warmup_epochs + 1)))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
