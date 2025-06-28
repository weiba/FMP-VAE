import os
import numpy as np
import torch
import torchvision
import argparse
from modules import ae, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
import copy
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import KFold
from dataloader import *

def inference(loader, model, device):
    model.eval()
    cluster_vector = []
    feature_vector = []
    for step, x in enumerate(loader):
        x = x.float().to(device)
        with torch.no_grad():
            c, h = model.forward_cluster(x)
        c = c.detach()
        h = h.detach()
        cluster_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())
    cluster_vector = np.array(cluster_vector)
    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return cluster_vector, feature_vector

def train(train_loader, device, model, optimizer, args):
    """
    在给定的训练集数据加载器上训练模型
    """
    loss_epoch = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_loss = 0
        for step, x in enumerate(train_loader):
            optimizer.zero_grad()
            x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
            x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            batch = x_i.shape[0]
            criterion_instance = contrastive_loss.DCL(temperature=0.5, weight_fn=None)
            criterion_cluster = contrastive_loss.ClusterLoss(args.cluster_number, args.cluster_temperature, loss_device).to(loss_device)
            loss_instance = criterion_instance(z_i, z_j) + criterion_instance(z_j, z_i)
            loss_cluster = criterion_cluster(c_i, c_j)
            loss = loss_instance + loss_cluster
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_epoch += epoch_loss
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {epoch_loss}")
    return loss_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--cluster_number', type=int, default=3)  # gbmnew
    parser.add_argument('--cluster_number', type=int, default=4)  # gbmnew

    args = parser.parse_args()

    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    model_path = './save'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载全部数据
    all_data = get_feature_my(args.batch_size, True).dataset.data
    data_size = len(all_data)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    dbi_scores = []
    si_scores = []
    all_fold_results = []  # 用于存储每一折的详细结果
    for cross_val_iter in range(10):  # 十次五折交叉验证
        print(f"Starting {cross_val_iter + 1}-th round of 5-fold cross-validation...")
        fold_dbi_scores = []
        fold_si_scores = []
        fold_results = []  # 存储当前轮次每一折的结果
        for fold, (train_index, test_index) in enumerate(kf.split(all_data)):
            print(f"Starting training for fold {fold + 1} in this round...")
            train_data = all_data[train_index]
            test_data = all_data[test_index]
            train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            test_loader = data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

            # 初始化模型相关变量
            ae_ = ae.AE()
            model = network.Network(ae_, args.feature_dim, args.cluster_number)
            model = model.to(device)

            # 优化器 / 损失
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            loss_device = device

            # 训练模型
            loss_epoch = train(train_loader, device, model, optimizer, args)

            # 推理
            model.eval()
            X, h = inference(test_loader, model, device)

            # 对X进行维度处理，使其满足评估指标计算要求
            if len(X.shape) == 1:
                num_samples = X.shape[0]
                new_X = np.zeros((num_samples, args.cluster_number))
                for i in range(num_samples):
                    new_X[i, X[i]] = 1
                X = new_X
            elif X.shape[1]!= args.cluster_number:
                X = np.hstack([1 - np.sum(X, axis=1, keepdims=True), X])

            print('-----------x----------')
            print(X)
            # 计算DBI和SI指标
            dbi = davies_bouldin_score(h, np.argmax(X, axis=1))
            si = silhouette_score(h, np.argmax(X, axis=1))
            fold_dbi_scores.append(dbi)
            fold_si_scores.append(si)
            fold_results.append([fold + 1, dbi, si])  # 将每一折的序号、DBI、SI添加到当前轮次的结果列表
            print(f"Fold {fold + 1} completed in this round. DBI: {dbi}, SI: {si}")
        print(f"5-fold cross-validation round {cross_val_iter + 1} completed.")
        dbi_scores.append(np.mean(fold_dbi_scores))
        si_scores.append(np.mean(fold_si_scores))
        all_fold_results.extend(fold_results)  # 将当前轮次的结果添加到总的结果列表

    # 计算平均结果和标准差
    mean_dbi = np.mean(dbi_scores)
    std_dbi = np.std(dbi_scores)
    mean_si = np.mean(si_scores)
    std_si = np.std(si_scores)

    print(f"DBI mean: {mean_dbi}, DBI std: {std_dbi}")
    print(f"SI mean: {mean_si}, SI std: {std_si}")

    # 将详细结果整理成DataFrame并输出
    columns = ["Fold", "DBI", "SI"]
    all_fold_df = pd.DataFrame(all_fold_results, columns=columns)
    print("\nDetailed Results for Each Fold in All Rounds:")
    print(all_fold_df)