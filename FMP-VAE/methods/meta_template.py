import torch.nn as nn
import numpy as np
from abc import abstractmethod
from utils.utils import *

def sim_combine(support):
        
        mean_samples = []
        support_tensor = []
        
        for i in range(support.shape[1]):
         
          tensor = support[i, :, :]
          
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

class MetaTemplate(nn.Module):
    def __init__(self,model_func, n_way, n_support, use_cuda=True, adaptation=False):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way  # N, n_classes
        self.n_support = n_support  # S, sample num of support set
        self.n_query = -1  # Q, sample num of query set(change depends on input)
        self.feature_extractor = model_func()  # feature extractor
        # self.feat_dim = self.feature_extractor.final_feat_dim
        # self.verbose = verbose
        self.use_cuda = use_cuda
        self.adaptation = adaptation

    def set_forward(self, x):
        z_support, z_query = self.parse_feature(x)
        #z_support = self.projection_mlp_1(z_support)
        #z_query = self.projection_mlp_1(z_query)
        #z_proto = sim_combine(z_support)
        z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
        if self.classification_head == 'consine':
            return self.cosine_similarity(z_query, z_proto) * 10
        else:
            return -self.euclidean_dist(z_query, z_proto)
            
    def parse_feature(self, x):
        x = x.requires_grad_(True)
        x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        x = x.float()
        print(x.shape)
        z_all = self.feature_extractor.forward(x)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
        z_support = z_all[:, :self.n_support]  # [N, S, d]
        z_query = z_all[:, self.n_support:]  # [N, Q, d]
        return z_support, z_query
        
    def correct(self, x):
        if self.adaptation:
            scores = self.set_forward_adaptation(x)
        else:
            scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
        topk_ind = topk_labels.cpu().numpy()  # index of topk
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        for i, (x, _, Me, Mi) in enumerate(train_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            print("loss",loss)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss

    def testloop(self, test_loader, record=None, return_std=False):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _, Me, Mi) in enumerate(test_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
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