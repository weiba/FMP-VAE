import torch
from data.datamgr import SetDataManager
from . import backbone
import os
import glob
import scipy.io as sio
import numpy as np
from sklearn.model_selection import KFold
import sklearn.preprocessing as sp
import random
import torch.nn.functional as F
#from methods.CSS import *

base_path = os.path.dirname(__file__).replace('\\', '/') + '/..'

model_dict = dict(
    AE=backbone.AE,
    VAE=backbone.VAE,
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101
)

if torch.cuda.is_available():
    use_cuda = True
    print('GPU detected, running with GPU!')
else:
    print('GPU not detected, running with CPU!')
    use_cuda = False
    
def sum_combine(supprot, query):
    for i in range(support.shape[1]):
      support = support[i, :, :]     
      query = query[i, :, :]
      score = scores = cosine_similarity(query, proto) 
      print(support.shape)
      print(query.shape)
      print(score.shape)
        
    pass
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

def get_train_loader(base_data, val_data, n_base_query, n_val_query, train_n_way, test_n_way, n_shot, num_workers=0):
    # n_query = max(1, int(
    #     6 * test_n_way / train_n_way))  # if test_n_way <train_n_way, reduce n_query to keep batch size small
    base_datamgr = SetDataManager(n_query=n_base_query, n_way=train_n_way, n_support=n_shot,
                                  num_workers=num_workers, n_eposide=50)  # n_eposide=100
    base_loader = base_datamgr.get_data_loader(base_data)

    val_datamgr = SetDataManager(n_query=n_val_query, n_way=test_n_way, n_support=n_shot,
                                  num_workers=num_workers, n_eposide=50)
    val_loader = val_datamgr.get_data_loader(val_data)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor
    return base_loader, val_loader


    
def get_test_loader(test_data, n_val_query, test_n_way, n_shot, num_workers=0):
    val_datamgr = SetDataManager(n_query=n_val_query, n_way=test_n_way, n_support=n_shot,
                                  num_workers=num_workers, n_eposide=50)
    test_loader = val_datamgr.get_data_loader(test_data)
    return test_loader
    
def get_checkpoint_dir(data_name,  train_n_way, n_shot, addition=None):
    if addition is None:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s' % (data_name)
    else:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s_%s' % (data_name, str(addition))
    if not algorithm in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (train_n_way, n_shot)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir

def get_data(data_name,Methy_features,Mirna_features, features_table_name, lables_table_name):
    data = sio.loadmat(data_name + '.mat')
    meta_data = data[features_table_name].T
    Methy_features = data[Methy_features].T
    Mirna_features = data[Mirna_features].T
    targets = torch.LongTensor(data[lables_table_name])
    cl_list = np.unique(targets).tolist()
    meta_data = torch.FloatTensor(sp.scale(meta_data))
    Methy_features=torch.FloatTensor(sp.scale(Methy_features))
    Mirna_features=torch.FloatTensor(sp.scale(Mirna_features))
    sub_meta_idx = {}
    print('cl',cl_list)
    for cl in cl_list:
        sub_meta_idx[cl] = []
    for i, x in enumerate(targets):
        sub_meta_idx[int(x[0])].append(i)
    kf_list = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for x in range(len(sub_meta_idx)):
        kf_list.append(kf.split(np.array(sub_meta_idx[x])))
    return cl_list, meta_data, sub_meta_idx, kf_list,Methy_features,Mirna_features

def get_train_data(data):
    list = data[2]
    for i in range(4):
        random.shuffle(list[i])
    base_dict = {}
    val_dict = {}
    for i in range(4):
        a = 4 * (len(list[i]) // 5)
        b = len(list[i])-a
        base = list[i][:a]
        val = list[i][a:]
        base_dict[i] = base
        val_dict[i] = val
    base_data = (data[0], data[1], base_dict, data[3], data[4])
    val_data = (data[0], data[1], val_dict, data[3], data[4])
    return base_data, val_data

def get_all_data(cl_list, meta_data, sub_meta_idx, kf_list,Methy_features,Mirna_features):
    tr = []
    te = []
    for j in kf_list:
        data = next(j)
        tr.append(data[0])
        te.append(data[1])
    base_sub_meta_idx = {}
    val_sub_meta_idx = {}
    for cl in cl_list:
        base_sub_meta_idx[cl] = []
        val_sub_meta_idx[cl] = []
    for cl in cl_list:
        a = sub_meta_idx[cl]
        for x in tr[cl]:
            base_sub_meta_idx[cl].append(a[x])
        for y in te[cl]:
            val_sub_meta_idx[cl].append(a[y])
    base_data = (cl_list, meta_data, base_sub_meta_idx,Methy_features,Mirna_features)
    val_data = (cl_list, meta_data, val_sub_meta_idx,Methy_features,Mirna_features)
    return base_data, val_data
base_path = os.path.dirname(__file__).replace('\\', '/') + '/..'

def get_resume_file(checkpoint_dir, epoch=None):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None
    if epoch is not None:
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
        return resume_file
    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

def get_checkpoint_dir(algorithm, model_name, dataset, train_n_way, n_shot, addition=None):
    if addition is None:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s' % (dataset, model_name, algorithm)
    else:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s_%s'%(dataset, model_name, str(addition))
    checkpoint_dir += '_%dway_%dshot' % (train_n_way, n_shot)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir

def get_novel_file(dataset, split='novel'):
    if dataset == 'cross':
        if split == 'base':
            loadfile = data_dir['miniImagenet'] + 'all.json'
        else:
            loadfile = data_dir['CUB'] + split + '.json'
    elif dataset == 'cross_char':
        if split == 'base':
            loadfile = data_dir['omniglot'] + 'noLatin.json'
        else:
            loadfile = data_dir['emnist'] + split + '.json'
    else:
        loadfile = data_dir[dataset] + split + '.json'
    return loadfile