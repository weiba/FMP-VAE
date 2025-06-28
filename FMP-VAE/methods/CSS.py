import torch
from torch import nn
from methods.meta_template import MetaTemplate
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import copy
import sklearn.preprocessing as sp
import warnings
from utils.utils import *
import pandas as pd
f_name = './results/'

warnings.filterwarnings("ignore")

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


def sim(z1, z2):
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)
    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t())
    sim_matrix = torch.exp(dot_numerator / dot_denominator / 0.5)
    return sim_matrix

def pro_contrastive_loss(h1, h2, tau):
    '''
    sim_matrix = sim(h1, h2) 
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()'''
    sim_matrix = sim(h1, h2)
    #print(sim_matrix.shape)
    numerator=torch.sum(sim_matrix[:10, 0])
    denominator=torch.sum(sim_matrix[10:, 0])

    l1 = -torch.log(numerator / denominator).mean()
    numerator = torch.sum(sim_matrix[10:21, 1])
    denominator = torch.sum(sim_matrix[:10, 1])+torch.sum(sim_matrix[20:, 1])

    l2 = -torch.log(numerator / denominator).mean()
    numerator = torch.sum(sim_matrix[20:31, 2])
    denominator = torch.sum(sim_matrix[:20, 2])+torch.sum(sim_matrix[30:, 2])
    
    l3 = -torch.log(numerator / denominator).mean()
    numerator = torch.sum(sim_matrix[-10:, 3])
    denominator = torch.sum(sim_matrix[:30, 3])
    l4 = -torch.log(numerator / denominator).mean()
    #sim_matrix2= sim_matrix.t()
    #sim_matrix = sim_matrix/(torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
    #lori_mp1 = -torch.log(sim_matrix.mul(pos).sum(dim=-1)).mean()
    return l1+l2+l3+l4

def prototype_contrastive_loss(samples, proto, labels, tau=0.5):
    """
    samples: Tensor of size [16, 128], representing 16 samples.
    proto: Tensor of size [4, 128], representing 4 prototypes.
    labels: Tensor of size [16], containing the class labels for each sample.
    tau: A temperature parameter controlling the softness of the softmax distribution.
    """
    # 计算样本与原型之间的相似度
    similarity = torch.matmul(samples, proto.T) / tau  # 形状为 [16, 4]

    # 使用softmax计算概率分布
    probs = F.softmax(similarity, dim=-1)  # 形状为 [16, 4]

    # 生成目标分布，只有正确类别的位置为1，其他为0
    targets = F.one_hot(labels, num_classes=proto.size(0)).float()  # 形状为 [16, 4]

    # 计算对比损失，这里使用的是交叉熵损失
    loss = F.cross_entropy(probs, labels)

    return loss
    
def ins_contrastive_loss(h1, h2, tau):

    sim_matrix = sim(h1, h2)
    #print(sim_matrix.shape)
    numerator=torch.sum(sim_matrix[:, :4][:10, :])
    denominator=torch.sum(sim_matrix[:, :4][10:, :])
    l1 = -torch.log((numerator/tau) / (denominator/tau)).mean()
    
    numerator = torch.sum(sim_matrix[:, 4:8][10:21, :])
    denominator = torch.sum(sim_matrix[:, 4:8][:10, :])+torch.sum(sim_matrix[:, 4:8][20:, :])
    l2 = -torch.log((numerator/tau) / (denominator/tau)).mean()
    
    numerator = torch.sum(sim_matrix[:, 9:12][20:31, :])
    denominator = torch.sum(sim_matrix[:, 9:12][:20, :])+torch.sum(sim_matrix[:, 9:12][30:, :]) 
    l3 = -torch.log((numerator/tau) / (denominator/tau)).mean()
    
    numerator = torch.sum(sim_matrix[:, :-4][-10:, :])
    denominator = torch.sum(sim_matrix[:, :-4][:30, :])
    l4 = -torch.log((numerator/tau) / (denominator/tau)).mean()
    #sim_matrix2= sim_matrix.t()
    #sim_matrix = sim_matrix/(torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
    #lori_mp1 = -torch.log(sim_matrix.mul(pos).sum(dim=-1)).mean()
    return l1+l2+l3+l4


def sim_combine(support):
        
        mean_samples = []
        support_tensor = []
        
        for i in range(support.shape[1]):
          #print(support.shape)
          tensor = support[i, :, :]
          #print(tensor.shape)
          similarities = F.cosine_similarity(tensor.unsqueeze(1), tensor.unsqueeze(0), dim=2)
      
          # 忽略对角线上的相似性（每个样本与自身的相似性为1）
          torch.diagonal(similarities).fill_(-1)
      
          # 找到最相似的两个样本的索引
          max_similarity_index = torch.argmax(similarities)
          sample_1_index = max_similarity_index // similarities.shape[1]
          sample_2_index = max_similarity_index % similarities.shape[1]
      
          # 计算最相似的两个样本的平均值
          mean_sample = torch.mean(torch.stack([tensor[sample_1_index], tensor[sample_2_index]], dim=0), dim=0)
      
          # 移除已求平均值的样本
          tensor = torch.cat(
              [tensor[:sample_1_index], tensor[sample_1_index + 1:sample_2_index], tensor[sample_2_index + 1:]], dim=0)
      
          # 合并最相似的两个样本的平均值与剩余样本
          tensor = torch.cat([tensor, mean_sample.unsqueeze(0)], dim=0)
          
          # 输出结果
    
          similarities = F.cosine_similarity(tensor.unsqueeze(1), tensor.unsqueeze(0), dim=2)
      
          # 忽略对角线上的相似性（每个样本与自身的相似性为1）
          torch.diagonal(similarities).fill_(-1)
      
          # 找到最相似的两个样本的索引
          max_similarity_index = torch.argmax(similarities)
          sample_1_index = max_similarity_index // similarities.shape[1]
          sample_2_index = max_similarity_index % similarities.shape[1]
      
          # 计算最相似的两个样本的平均值
          mean_sample = torch.mean(torch.stack([tensor[sample_1_index], tensor[sample_2_index]], dim=0), dim=0)
      
          # 移除已求平均值的样本
          tensor = torch.cat(
              [tensor[:sample_1_index], tensor[sample_1_index + 1:sample_2_index], tensor[sample_2_index + 1:]], dim=0)
      
          # 合并最相似的两个样本的平均值与剩余样本
          merged_tensor = torch.cat([tensor, mean_sample.unsqueeze(0)], dim=0)
          merged_tensor = torch.mean(merged_tensor,dim=0)
      
          support_tensor.append(merged_tensor)
          
        support_tensor = torch.stack(support_tensor,dim=0)  
          
        return support_tensor    

class LN(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, use_cuda=True, verbose=False, classification_head='cosine'):
        super(LN, self).__init__(model_func, n_way, n_support, use_cuda=use_cuda)
        self.loss_fn = nn.CrossEntropyLoss()
        self.ssl_loss = nn.CosineEmbeddingLoss()
        self.verbose = verbose
        self.vae = model_func()
        self.classification_head = classification_head
        self.ssl_feature_extractor = copy.deepcopy(self.feature_extractor)
        self.mlp_0 = nn.Sequential(
            nn.Linear(128, 1024)
        )
        
        self.mlp_1 = nn.Sequential(
            nn.Linear(20531, 1024),
        )
        self.mlp_2 = nn.Sequential(
           nn.Linear(25531, 1024),
        )
        self.mlp_3 = nn.Sequential(
            nn.Linear(21577, 1024),
        )
        
        self.mlp_4 = nn.Sequential(
            nn.Linear(128, 4),
        )
        '''
        self.mlp_1 = nn.Sequential(
            nn.Linear(12042, 1024),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(17042, 1024),
        )
        self.mlp_3 = nn.Sequential(
            nn.Linear(12576, 1024),
        )
        '''
        self.projection_mlp_2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024)
        )
        self.prediction_mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1024),
        )
        self.alpha = nn.Parameter(torch.ones([1]))
        self.gamma = nn.Parameter(torch.ones([1]) * 2, requires_grad=False)
        self.net = model_func
        # self.pre = nn.Sequential(
        #     nn.Linear(20531, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 4),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        output = self.pre(x)
        return output
        
    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def pre_train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        for i, (x, _, Me, Mi) in enumerate(train_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, 20531]
            self.n_way = x.size(0)
            optimizer.zero_grad()
            x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            '''
            x = self.mlp_1(x)
            print(x.shape)
            recon_batch, mu, logvar = self.feature_extractor.forward(x)
            print('111111',recon_batch.shape)
            print('222222',mu.shape)
            print('333333',logvar.shape)
            loss1 = self.loss_function(recon_batch, x, mu, logvar)
            print()
            z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
            z_support = z_all[:, :self.n_support]  # [N, S, d]
            z_query = z_all[:, self.n_support:]  # [N, Q, d]
            
            #z_support, z_query = self.parse_feature(x)
            z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
            z_query = z_query.reshape(self.n_way * self.n_query, -1)
            z_support = z_support.reshape(self.n_way * self.n_support, -1)
            
           # loss_pro = pro_contrastive_loss(z_query, z_proto, 0.1)
            #loss_ins = ins_contrastive_loss(z_query,  z_support, 0.1)
           # cls_socre = self.mlp_4(z_query)
           # cls_socre = F.softmax(cls_socre, dim=0)
            
            loss2 = self.cosine_similarity(z_query, z_proto) * 10
            

            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
            if self.use_cuda:         
                y_query = y_query.cuda()
            loss_cls = self.loss_fn(cls_socre,y_query)
            loss = loss_pro+loss_ins
            
            x = self.mlp_1(x)
            z_support, z_query = self.parse_feature(x)
            z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
            z_query = z_query.reshape(self.n_way * self.n_query, -1)
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
            if self.use_cuda:         
                y_query = y_query.cuda()
            loss = prototype_contrastive_loss(z_query,z_proto,y_query)
            
            loss = loss1+loss2
            '''
            a = self.feature_extractor.forward(x)
            z_all = self.feature_extractor.encode(x)
            print(a)
            print(a.shape)
            print(z_all)
            print(z_all.shape)
            z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])
            z_support = z_all[:, :self.n_support]  # [N, S, d]
            z_query = z_all[:, self.n_support:] 
            z_query = z_query.reshape(self.n_way * self.n_query, -1)
            z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)
            z_query = z_query.reshape(self.n_way * self.n_query, -1)
            print(z_query)
            print(z_query.shape)
            print(z_proto)
            print(z_proto.shape)
            loss = self.cosine_similarity(z_query, z_proto) * 10+criterion(outputs, targets)
            

            
        
            
            
            
            
            
            
            #loss = self.set_pre_train_forward_loss(x, Me, Mi)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss

    def set_pre_train_forward_loss(self, x, Me, Mi):
        #y_query = query[:, :10].reshape(-1).long()
        #y_query = torch.from_numpy(np.repeat([query[0][0],[1][0]], self.n_query)).long()
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
      
        if self.use_cuda:
            #y_query = y_query.cuda()
            y = y.cuda()
    
        x = self.mlp_1(x)
        scores , z_query= self.set_pre_train_forward(x)
        
        #cls_socre = self.mlp_4(z_query)
        #cls_socre = F.softmax(cls_socre, dim=0)
        #print(y_query)
        #print(y)
        #print(scores)
        #scores = self.set_pre_train_sim_forward(x)
        return self.loss_fn(scores, y)#+self.loss_fn(cls_socre,y)
        

    def set_pre_train_forward(self, x):
        z_support, z_query = self.parse_feature(x)
        
        #z_support = self.projection_mlp_1(z_support)
        #z_query = self.projection_mlp_1(z_query)
        #z_proto = sim_combine(z_support)
        z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
        if self.classification_head == 'cosine':
            return self.cosine_similarity(z_query, z_proto) * 10,z_query
        else:
            return -self.euclidean_dist(z_query, z_proto)

    # 单组学！
    def pre_train_test_loop(self, test_loader, record=None, return_std=False):
        acc_all = 0
        # r_all = []
        iter_num = len(test_loader)
        p_all=0
        r_all=0
        F1_all=0
        count = 0
        ARI_ALL=MCC_ALL=DB_ALL=SI_ALL=0
        warnings.filterwarnings("ignore")
        for i, (x, T, Me, Mi) in enumerate(test_loader):
            
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            with torch.no_grad():
                x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
                x = self.mlp_1(x)
                z_all = self.feature_extractor.forward(x)#特征提取器！！！！
                z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
                z_support = z_all [:, :self.n_support]  # [N, S, d]
                z_query = z_all [:, self.n_support:]  # [N, Q, d]
                #z_proto = sim_combine(z_support)
                #z_proto = sum_combine(z_support, z_query)
                z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
    
                z_query = z_query.reshape(self.n_way * self.n_query, -1)
                  # [N*Q,d]
                scores = self.cosine_similarity(z_query, z_proto)
                
                #print("+++++", scores.shape)
                #y_query = _[:, :6].reshape(-1).long()
                y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
                #y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
                #y = y.cuda()
                #topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
                #topk_ind = topk_labels.cpu().numpy()  # index of topk
                #acc_all.append(np.sum(topk_ind[:, 0] == y_query) / len(y_query) * 100)
                _, a = scores.max(dim=1)
               
                b = scores.cpu().numpy()
                a = a.cpu().numpy()
                #print(y_query)
                #print(a)
                # p_all = metrics.precision_score(y_query, a, average="macro")
                # p_w = metrics.precision_score(y_query, a, average="weighted")
                p = metrics.precision_score(y_query, a, average=None)
                p_all = p+p_all
                # print("Precision_all: ", p_all)
                # print("p_weighted: ", p_w)
                # print("Precision: ", p)
                #
                ACC = metrics.accuracy_score(y_query, a)
                # ACC = metrics.accuracy_score(y_query, a, normalize=False)
                # print("acc_all: ", acc_all)
                # print("ACC_all: ", ACC_all)
                # print("ACC: ", ACC)
                acc_all=acc_all+ACC
                #
                # r_all = metrics.recall_score(y_query, a, average="macro")
                # r_w = metrics.recall_score(y_query, a, average="weighted")
                r = metrics.recall_score(y_query, a, average=None)
                r_all = r + r_all
                # print("recall_all: ", r_all)
                # print("recall_w: ", r_w)
                # print("recall: ", r)
                #
                # F1_all = metrics.f1_score(y_query, a, average="macro")
                # F1_w = metrics.f1_score(y_query, a, average="weighted")
                F1 = metrics.f1_score(y_query, a, average=None)
                F1_all = F1+F1_all
                # print("F1_all: ", F1_all)
                # print("F1_w: ", F1_w)
                # print("F1: ", F1)
                #
                ARI = metrics.adjusted_rand_score(y_query, a)
                ARI_ALL = ARI+ARI_ALL
                # print("ARI: ", ARI)
                #
                MCC = metrics.matthews_corrcoef(y_query, a)
                MCC_ALL = MCC+MCC_ALL
                # print("MCC: ", MCC)
                #
                DB= metrics.davies_bouldin_score(b, a)
                DB_ALL =DB+DB_ALL
                # print("DB: ", DB)
                #
                SI = metrics.silhouette_score(b, a)
                SI_ALL =SI+ SI_ALL
                # print("Si: ", Si)
                #
                # name = f_name + "DB" + '_' +'result' + 'txt'
                # f = open(name, 'a')
                # print('DB : ', DB, file=f)
                # f.close()
                count =  count +1 
        acc_mean = acc_all/ count
        p=p_all/ count
        r=r_all/ count
        F1=F1_all/ count
        ARI=ARI_ALL/ count
        MCC=MCC_ALL/ count
        DB=DB_ALL/ count
        SI=SI_ALL/ count        
        return acc_mean,p,r,F1,ARI,MCC,DB,SI

    def pre (self, x, record=None, return_std=False):
        x = self.mlp_1(x)
        z_all = self.feature_extractor.forward(x)

#单组学！
    def pre_traintest_loop(self, test_loader, record=None, return_std=False):
        acc_all = 0
        # r_all = []
        iter_num = len(test_loader)
        p_all=0
        r_all=0
        F1_all=0
        count = 0
        ARI_ALL=MCC_ALL=DB_ALL=SI_ALL=0
        warnings.filterwarnings("ignore")
        for i, (x, T, Me, Mi) in enumerate(test_loader):
            
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            with torch.no_grad():
                x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
                x = self.mlp_1(x)
                z_all = self.feature_extractor.forward(x)
                z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
                z_support = z_all [:, :self.n_support]  # [N, S, d]
                z_query = z_all [:, self.n_support:]  # [N, Q, d]
                #z_proto = sim_combine(z_support)
                #z_proto = sum_combine(z_support, z_query)
                z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
    
                z_query = z_query.reshape(self.n_way * self.n_query, -1)
                  # [N*Q,d]
                scores = self.cosine_similarity(z_query, z_proto)
                
                #print("+++++", scores.shape)
                #y_query = _[:, :6].reshape(-1).long()
                y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
                #y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
                #y = y.cuda()
                #topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
                #topk_ind = topk_labels.cpu().numpy()  # index of topk
                #acc_all.append(np.sum(topk_ind[:, 0] == y_query) / len(y_query) * 100)
                _, a = scores.max(dim=1)
               
                b = scores.cpu().numpy()
                a = a.cpu().numpy()
                #print(y_query)
                #print(a)
                # p_all = metrics.precision_score(y_query, a, average="macro")
                # p_w = metrics.precision_score(y_query, a, average="weighted")
                p = metrics.precision_score(y_query, a, average=None)
                p_all = p+p_all
                # print("Precision_all: ", p_all)
                # print("p_weighted: ", p_w)
                # print("Precision: ", p)
                #
                ACC = metrics.accuracy_score(y_query, a)
                # ACC = metrics.accuracy_score(y_query, a, normalize=False)
                # print("acc_all: ", acc_all)
                # print("ACC_all: ", ACC_all)
                # print("ACC: ", ACC)
                acc_all=acc_all+ACC
                #
                # r_all = metrics.recall_score(y_query, a, average="macro")
                # r_w = metrics.recall_score(y_query, a, average="weighted")
                r = metrics.recall_score(y_query, a, average=None)
                r_all = r + r_all
                # print("recall_all: ", r_all)
                # print("recall_w: ", r_w)
                # print("recall: ", r)
                #
                # F1_all = metrics.f1_score(y_query, a, average="macro")
                # F1_w = metrics.f1_score(y_query, a, average="weighted")
                F1 = metrics.f1_score(y_query, a, average=None)
                F1_all = F1+F1_all
                # print("F1_all: ", F1_all)
                # print("F1_w: ", F1_w)
                # print("F1: ", F1)
                #
                ARI = metrics.adjusted_rand_score(y_query, a)
                ARI_ALL = ARI+ARI_ALL
                # print("ARI: ", ARI)
                #
                MCC = metrics.matthews_corrcoef(y_query, a)
                MCC_ALL = MCC+MCC_ALL
                # print("MCC: ", MCC)
                #
                DB= metrics.davies_bouldin_score(b, a)
                DB_ALL =DB+DB_ALL
                # print("DB: ", DB)
                #
                SI = metrics.silhouette_score(b, a)
                SI_ALL =SI+ SI_ALL
                # print("Si: ", Si)
                #
                # name = f_name + "DB" + '_' +'result' + 'txt'
                # f = open(name, 'a')
                # print('DB : ', DB, file=f)
                # f.close()
                if ACC <0.71:
                  df = pd.DataFrame(T)

# 保存DataFrame到Excel文件中
                  excel_path = './results/table.xlsx'

                  
                  # 尝试读取现有的Excel文件，如果文件不存在，则创建一个空的DataFrame
                  try:
                      existing_df = pd.read_excel(excel_path, header=None)
                  except FileNotFoundError:
                      existing_df = pd.DataFrame()
                  
                  # 创建一个新的DataFrame来存储新的张量数据
                  new_df = pd.DataFrame(T)
                  
                  # 将新的数据追加到现有的DataFrame的末尾
                  updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                  
                  # 将更新后的DataFrame保存回原Excel文件，覆盖原有数据
                  updated_df.to_excel(excel_path, index=False, header=False)
                  #name = './results/' + "table2" + '.txt'
                  #f = open(name, 'a')
                  #print('Bad : ',str(T), file=f)
                  #f.close()
                  print('ACC:',ACC)
                count =  count +1 
        acc_mean = acc_all/ count
        p=p_all/ count
        r=r_all/ count
        F1=F1_all/ count
        ARI=ARI_ALL/ count
        MCC=MCC_ALL/ count
        DB=DB_ALL/ count
        SI=SI_ALL/ count        
        return acc_mean,p,r,F1,ARI,MCC,DB,SI



    def euclidean_dist(self, x, y): #欧式距离
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return torch.pow(x - y, 2).sum(2)

    def cosine_similarity(self, x, y): #余弦相似度
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return (x * y).sum(2)

    def set_pre_train_sim_forward(self, x):
        z_support, z_query = self.parse_feature(x)
        z_query = z_query.reshape(self.n_way * self.n_query, -1)
        averages = []
        # 遍历每个类别的支持样本
        for i in range(self.n_way):
            # 获取当前类别的支持样本
            support = z_support[i]
            
            # 将平均值添加到列表中
            averages.append(self.cosine_similarity(z_query, support))
        averages = torch.stack(averages,dim=0)
        aver_score = averages
        return aver_score

    def f(self, x):
        # x:[N*(S+Q),n_channel,h,w]
        #x = self.projection_mlp_0(x)
        x = self.ssl_feature_extractor(x)
        x = self.mlp_0(x)
        #x = self.projection_mlp_1(x)
        x = self.projection_mlp_2(x)
        return x

    def h(self, x):
        x = self.prediction_mlp(x)
        return x

    def D(self, p, z):
        z = z.detach()
        p = torch.nn.functional.normalize(p, dim=1)
        z = torch.nn.functional.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def data_augmentation_1(self, img, seed):
        # x:torch.Tensor
        img = torch.FloatTensor(img)
        np.random.seed(seed)
        x = np.random.choice(img, size=15000, replace=False, p=None)
        x = torch.FloatTensor(x)
        return x

    def data_augmentation_2(self, img):
        # x:torch.Tensor
        img = torch.FloatTensor(img)
        x = img[:15000]
        return x

    def data_augmentation_3(self, img):
        # x:torch.Tensor
        img = torch.FloatTensor(img)
        x = img[-15000:]
        return x

    def contrastive_loss(self, Me, Mi):  # 对比损失
        # x:[N*(S+Q),n_channel,h,w]    x=[80,20531]
        #x = x.cpu()
        #x1 = F.dropout(x, 0.2)
        #x2 = F.dropout(x, 0.2)
        # a = len(x)
        # x1 = torch.empty((len(x), 15000))
        # x2 = torch.empty((len(x), 15000))
        # for index in range(x.shape[0]):
        #     # x1[index] = self.data_augmentation_1(x[index], 84)
        #     # x2[index] = self.data_augmentation_1(x[index], 48)
        #     x1[index] = self.data_augmentation_2(x[index])
        #     x2[index] = self.data_augmentation_3(x[index])
        # x1 = torch.FloatTensor(x1).cuda()
        # x2 = torch.FloatTensor(x2).cuda()
        z1, z2 = self.f(Me), self.f(Mi)
        p1, p2 = self.h(Me), self.h(Mi)
        loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        return loss

    def ssl_train_loop(self, epoch, train_loader, optimizer):
        self.train()
        print_freq = 10
        avg_loss = 0
        for i, (x, _, Me, Mi) in enumerate(train_loader):  # x:[N, S+Q, n_channel, h, w]
            y = torch.cat([torch.full((12,), i) for i in range(4)])
            if self.use_cuda:
                x = x.cuda()
                Me = Me.cuda()
                Mi = Mi.cuda()
                y = y.cuda()
            x = x.reshape([x.shape[0] * x.shape[1], *x.shape[2:]])  # x:[N*(S+Q),n_channel,h,w]
            Me = Me.reshape([Me.shape[0] * Me.shape[1], *Me.shape[2:]])
            Mi = Mi.reshape([Mi.shape[0] * Mi.shape[1], *Mi.shape[2:]])
            x_Me = torch.cat((x, Me), dim=1)
            x_Mi = torch.cat((x, Mi), dim=1)
            x = self.mlp_1(x)
            Me = self.mlp_2(x_Me)
            Mi = self.mlp_3(x_Mi)
            x_ssl = torch.nn.functional.normalize(self.ssl_feature_extractor(Me), dim=1)#因为dim=1，所以是对行操作。
            
            x_pre = torch.nn.functional.normalize(self.feature_extractor(Mi).detach(), dim=1)
  
            optimizer.zero_grad()
            
            #loss = self.contrastive_loss(Me, Mi) + torch.mean(torch.sum((x_ssl * x_pre), dim=1))
            loss =  torch.mean(torch.sum((x_ssl * x_pre), dim=1))
            #loss = self.ssl_loss(Me, Mi, y) + torch.mean(torch.sum((x_ssl * x_pre), dim=1))
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))

    def ssl_test_loop(self, test_loader, record=None, return_std=False):
        warnings.filterwarnings("ignore")
        acc_all = 0
        iter_num = len(test_loader)
        p_all=0
        r_all=0
        F1_all=0
        count =  0
        ARI_ALL=MCC_ALL=DB_ALL=SI_ALL=0
        for i, (x, _, Me, Mi) in enumerate(test_loader):
            if self.use_cuda:
                x = x.cuda()
                Me = Me.cuda()
                Mi = Mi.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            with torch.no_grad():
                x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
                x = self.mlp_1(x)
                z_all = self.ssl_feature_extractor.forward(x)                
                z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
                z_support = z_all[:, :self.n_support]  # [N, S, d]
                z_query = z_all[:, self.n_support:]  # [N, Q, d]
                #z_proto = sim_combine(z_support)
                z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
                z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
                scores = self.cosine_similarity(z_query, z_proto)
                y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
                #topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
                #topk_ind = topk_labels.cpu().numpy()  # index of topk
                #acc_all.append(np.sum(topk_ind[:, 0] == y_query) / len(y_query) * 100)
                _, a = scores.max(dim=1)
                b = scores.cpu().numpy()
                a = a.cpu().numpy()
                acc = metrics.accuracy_score(y_query, a)
                acc_all = acc+acc_all
                p = metrics.precision_score(y_query, a, average=None)
                p_all = p+p_all
                r = metrics.recall_score(y_query, a, average=None)
                r_all = r + r_all
                F1 = metrics.f1_score(y_query, a, average=None)
                F1_all = F1+F1_all
                ARI = metrics.adjusted_rand_score(y_query, a)
                ARI_ALL = ARI+ARI_ALL

                MCC = metrics.matthews_corrcoef(y_query, a)
                MCC_ALL = MCC+MCC_ALL

                DB= metrics.davies_bouldin_score(b, a)
                DB_ALL =DB+DB_ALL

                SI = metrics.silhouette_score(b, a)
                SI_ALL =SI+ SI_ALL
                count =  count +1 
        acc = acc_all/ count
        p=p_all/ count
        r=r_all/ count
        F1=F1_all/ count
        ARI=ARI_ALL/ count
        MCC=MCC_ALL/ count
        DB=DB_ALL/ count
        SI=SI_ALL/ count        
        return acc,p,r,F1,ARI,MCC,DB,SI
            
    def ssl_testloop(self, test_loader, record=None, return_std=False):
        acc_all = []
        iter_num = len(test_loader)
        p_all=0
        r_all=0
        F1_all=0
        for i, (x, _) in enumerate(test_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            #with torch.no_grad():
            x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.ssl_feature_extractor.forward(x)
            z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
            z_support = z_all[:, :self.n_support]  # [N, S, d]
            z_query = z_all[:, self.n_support:]  # [N, Q, d]
            z_proto = sim_combine(z_support)
            #z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
            z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
            scores = self.cosine_similarity(z_query, z_proto)
            y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
            topk_ind = topk_labels.cpu().numpy()  # index of topk
            acc_all.append(np.sum(topk_ind[:, 0] == y_query) / len(y_query) * 100)
            _, a = scores.max(dim=1)
            a = a.cpu().numpy()
            p = metrics.precision_score(y_query, a, average=None)
            p_all = p+p_all
            r = metrics.recall_score(y_query, a, average=None)
            r_all = r + r_all
            F1 = metrics.f1_score(y_query, a, average=None)
            F1_all = F1+F1_all
        p=p_all/50
        r=r_all/50
        F1=F1_all/50
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean#,p,r,F1
            
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
        if self.use_cuda:
            y_query = y_query.cuda()
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)

    def meta_train_loop(self, epoch, train_loader, optimizer):
        self.train()
        self.pre_feature_extractor = copy.deepcopy(self.feature_extractor)
        print_freq = 10
        avg_loss = 0
        for i, (x, _, Me, Mi) in enumerate(train_loader):  # x:[N, S+Q, n_channel, h, w]
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            if self.use_cuda:
                x = x.cuda()
            xx = x.reshape([x.shape[0] * x.shape[1], *x.shape[2:]])  # x:[N*(S+Q),n_channel,h,w]
            with torch.no_grad():
                xx = self.mlp_1(xx)
                x_pre = self.pre_feature_extractor(xx)
                x_ssl = self.ssl_feature_extractor(xx)
                x_aggregation = nn.functional.normalize(torch.cat([x_pre, x_ssl], dim=1), dim=1)
                similarity = torch.sum(torch.unsqueeze(x_aggregation, dim=0) * torch.unsqueeze(x_aggregation, dim=1),
                                       dim=2)
                for index in range(similarity.shape[0]):
                    similarity[index, index] = 0
                D = torch.diag(torch.sum(similarity, dim=1) ** -0.5)
                A = D @ similarity @ D
            if self.use_cuda:
                augment = (self.alpha * torch.eye(A.shape[0], A.shape[0]).cuda() + A) ** self.gamma
            else:
                augment = (self.alpha * torch.eye(A.shape[0], A.shape[0]) + A) ** self.gamma
            z_all = augment @ self.feature_extractor(xx)
            z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, -1)
            z_support = z_all[:, :self.n_support]
            z_query = z_all[:, self.n_support:]
            #z_support = self.projection_mlp_1(z_support)
            #z_query = self.projection_mlp_1(z_query)
            #z_proto = sim_combine(z_support)
            z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
            z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
            if self.use_cuda:
                y_query = y_query.cuda()
            scores = self.cosine_similarity(z_query, z_proto) * 10
            optimizer.zero_grad()
            x = self.mlp_1(x)
            loss = self.loss_fn(scores, y_query) + self.set_forward_loss(x)
            
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss
        
    def test_loop(self, test_loader, record=None, return_std=False):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _, Me, Mi) in enumerate(test_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            x = self.mlp_1(x)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('acc_mean',acc_mean)
        if self.verbose:
            # Confidence Interval   90% -> 1.645      95% -> 1.96     99% -> 2.576
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean