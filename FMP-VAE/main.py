import numpy as np
import pandas as pd
from utils.utils import *
from methods.CSS import LN
import torch
import sys
import os
from torch import nn
import warnings
from model import Variable_AutoEncoder
warnings.filterwarnings("ignore")

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

# Press the green button in the gutter to run the script.

# n_shot = int(sys.argv[1])
# train_n_way = int(sys.argv[2])
# test_n_way = int(sys.argv[2])
# n_shot = 7
# train_n_way = 3
# test_n_way = 3


def pre_train():
    print('Start pre-training!')
    print(model_dict[model_name])
    model = LN(model_dict[model_name], n_way=train_n_way, n_support=n_shot, use_cuda=use_cuda, classification_head=classification_head)
    if use_cuda:
        model = model.cuda()
    max_acc = 0
    max_r = 0
    max_p = 0
    max_F1 = 0
    optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 1e-6},
                                  {'params': model.mlp_1.parameters(), 'lr': 1e-6},
                                  {'params': model.mlp_4.parameters(), 'lr': 1e-6}])
    for pre_epoch in range(0, train_epoch):
        model.train()
        model.pre_train_loop(pre_epoch, base_loader, optimizer)  # model are called by reference, no need to return        model.eval()
        model.eval()
        acc,p,r,F1,ARI,MCC,DB,SI = model.pre_train_test_loop(val_loader)
        outfile = os.path.join(checkpoint_dir, 'pre_train_best.tar')
        torch.save({'epoch': pre_epoch, 'state': model.state_dict()}, outfile)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
            print('epoch:', pre_epoch, 'pre_train val acc:', acc, 'best!')
            max_acc = acc
            #max_p =p
            #max_r = r
            #max_F1 = F1 
            outfile = os.path.join(checkpoint_dir, 'pre_train_best.tar')
            torch.save({'epoch': pre_epoch, 'state': model.state_dict()}, outfile)
        if (pre_epoch % save_freq == 0) or (pre_epoch == stop_epoch - 1):
            outfile = os.path.join(checkpoint_dir, 'pre_train_{:d}.tar'.format(pre_epoch))
            torch.save({'epoch': pre_epoch, 'state': model.state_dict()}, outfile)
            name = './results/n.txt'
            f = open(name, "a")
            print('MCC : ', np.mean(max_acc), 'seed : ', np.mean(seed), file=f)
    return model,max_acc,max_r,max_p,max_F1

def ssl_train():
    print('Start ssl-training!')
    model = LN(model_dict[model_name], n_way=train_n_way, n_support=n_shot, use_cuda=use_cuda, classification_head=classification_head)
    if use_cuda:
        model = model.cuda()
    max_acc = 0
    max_r = 0
    max_p = 0
    max_F1 = 0
    #torch.backends.cudnn.benchmark = True
    outfile = os.path.join(checkpoint_dir, 'pre_train_best.tar')
    tmp = torch.load(outfile)
    model.load_state_dict(tmp['state'])
    #model = torch.nn.DataParallel(model)
    max_acc = 0
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer = torch.optim.Adam([{'params': model.ssl_feature_extractor.parameters(), 'lr': 1e-6},
                                  {'params': model.projection_mlp_2.parameters(), 'lr': 1e-6},
                                  {'params': model.mlp_2.parameters(), 'lr': 1e-6},
                                  {'params': model.mlp_3.parameters(), 'lr': 1e-6},
                                  {'params': model.prediction_mlp.parameters(), 'lr': 1e-6} ])
    for ssl_epoch in range(0, train_epoch):
        model.train()
        model.ssl_train_loop(ssl_epoch, base_loader,optimizer)
        model.eval()
        acc,p,r,F1,ARI,MCC,DB,SI = model.ssl_test_loop(val_loader)
        print('acc:',acc)
        outfile = os.path.join(checkpoint_dir, 'ssl_train_best.tar')
        torch.save({'epoch': ssl_epoch, 'state': model.state_dict()}, outfile)
        if acc > max_acc:
            print('epoch:', ssl_epoch, 'ssl_train val acc:', acc, 'best!')
            max_acc = acc
            #max_p =p
            #max_r = r
            #max_F1 = F1 
            outfile = os.path.join(checkpoint_dir, 'ssl_train_best.tar')
            torch.save({'epoch': ssl_epoch, 'state': model.state_dict()}, outfile)
        if (ssl_epoch % save_freq == 0) or (ssl_epoch == 100 - 1):
            outfile = os.path.join(checkpoint_dir, 'ssl_train_{:d}.tar'.format(ssl_epoch))
            torch.save({'epoch': ssl_epoch, 'state': model.state_dict()}, outfile)
    return model,max_acc,max_r,max_p,max_F1


def test(phase='test'):
    print('Start testing!')
    device = torch.device('cuda')
    model = LN(model_dict[model_name], n_way=train_n_way, n_support=n_shot, use_cuda=use_cuda, classification_head=classification_head)
    model = model.cuda()
    if phase == 'pre':
        modelfile = os.path.join(checkpoint_dir, 'pre_train_best.tar')
        assert modelfile is not None
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        #model = nn.DataParallel(model)
        model.to(device)
    elif phase == 'ssl':
        modelfile = os.path.join(checkpoint_dir, 'ssl_train_best.tar')
        assert modelfile is not None
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
    elif phase == 'test':
        modelfile = get_best_file(checkpoint_dir)
        assert modelfile is not None
        tmp = torch.load(modelfile)

        model.load_state_dict(tmp['state'],strict=False)
    
    #loadfile = get_novel_file(dataset=dataset, split='novel')
    test_loader = get_test_loader(test_data=test_data, n_val_query=n_val_query, test_n_way=test_n_way,
                                                 n_shot=n_shot, num_workers=0)
    model.eval()
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if phase == 'pre':
        acc_mean,p,r,F1,ARI,MCC,DB,SI = model.pre_traintest_loop(test_loader, return_std=True)
        name = f_name + "PRE_" + '_' +'result' + '.txt'
        f = open(name, "a")
        print('Precision : ', p, file=f)
        print('Recall : ', r, file=f)
        print('f1score : ', F1, file=f)
        print('ACC : ', acc_mean, file=f)
        print('ARI : ', ARI, file=f)
        print('MCC : ', MCC, file=f)
        #print('AUC : ', AUC, file=f)
        print('SI : ', SI, file=f)
        print('DBI : ', DB, file=f)
        f.close()
        name = f_name + "table2" + '.txt'
        f = open(name, 'a')
        print('_________', file=f)
        f.close()
    elif phase == 'ssl':
        acc_mean,p,r,F1,ARI,MCC,DB,SI = model.ssl_test_loop(test_loader, return_std=True)
        name = f_name + "SSL_" + '_' +'result' + '.txt'
        f = open(name, "a")
        print('222', file=f)
        print('Precision : ', p, file=f)
        print('Recall : ', r, file=f)
        print('f1score : ', F1, file=f)
        print('ACC : ', acc_mean, file=f)
        print('ARI : ', ARI, file=f)
        print('MCC : ', MCC, file=f)
        #print('AUC : ', AUC, file=f)
        print('SI : ', SI, file=f)
        print('DBI : ', DB, file=f)
        f.close()


if __name__ == '__main__':
    data_name = "brca"
    features_table_name = data_name+"_Gene_Expression"
    Methy_features = data_name+"_Methy_Expression"
    Mirna_features = data_name+"_Mirna_Expression"
    lables_table_name = data_name+"_clinicalMatrix"
    print(lables_table_name)
    n_shot = 4  # number of labeled data in each class, same as n_support
    train_n_way = 4
    test_n_way = 4
    n_base_query = 8
    n_val_query = 4
    train_epoch = 20
    save_freq = 10
    stop_epoch = -1
    classification_head = 'cosine'
    model_name = 'VAE'
    algorithm = 'css'
    f_name = './results/'
    test_iter_num=600
    if_pre_train = True
    if_test = True

    seed = 84
    torch.manual_seed(seed)

    checkpoint_dir = get_checkpoint_dir(algorithm=algorithm, model_name=model_name, dataset=data_name,
                                        train_n_way=train_n_way, n_shot=n_shot)
    cl_list, meta_data, sub_meta_idx, kf_list,Methy_features,Mirna_features = get_data(data_name=data_name, features_table_name=features_table_name,Methy_features=Methy_features,  Mirna_features =Mirna_features, lables_table_name=lables_table_name)
    preacc = sslacc = metaacc = 0
    prep = sslp = metap = 0
    prer = sslr = metar = 0
    preF = sslF = metaF = 0
    for i in range(5):
      seed = 84
      train_data, test_data = get_all_data(cl_list=cl_list, meta_data=meta_data, sub_meta_idx=sub_meta_idx, kf_list=kf_list,Methy_features=Methy_features,Mirna_features=Mirna_features)
      base_data, val_data = get_train_data(train_data)
      torch.seed()
      base_loader, val_loader = get_train_loader(base_data=base_data, val_data=val_data, n_base_query=n_base_query,
                                                 n_val_query=n_val_query, train_n_way=train_n_way, test_n_way=test_n_way,
                                                 n_shot=n_shot, num_workers=0)
      _, pre_acc,pre_r,pre_p,pre_F1 = pre_train()
      test('pre')
      _, ssl_acc,ssl_r,ssl_p,ssl_F1 = ssl_train()
      test('ssl')








