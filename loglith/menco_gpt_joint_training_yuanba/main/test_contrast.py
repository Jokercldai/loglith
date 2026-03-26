



import torch
import numpy as np
import os
import sys

from config import read_args
from data_loader import *
from model import *
from model_contrast import *
from utils import *
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/cldai/sciresearch/loglith")
from toolkit.plot_fig import *







def test_one_epoch(model, test_loader, criterion, device, num_class):
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds, all_targets, all_fea_ori, all_sam_md = [], [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = {key: tensor.to(device) for key, tensor in data.items()}
            # batch_x, target, fea_ori = data['fea'], data['lith'], data['fea_ori']
            batch_x, target, fea_ori = data['fea_log'], data['lith'], data['fea_ori']
            sam_md = data['sam_md']

            outputs = model(batch_x, model_task=args.stage)
            loss = criterion(outputs.view(-1, num_class), target.view(-1))
            test_loss += loss.item()

            preds = outputs.argmax(dim=-1)
            test_correct += (preds == target).sum().item()
            test_total += target.numel()

            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
            all_fea_ori.append(fea_ori.cpu())
            all_sam_md.append(sam_md.cpu())
            # print('preds.cpu().shape, fea_ori.cpu().shape \n', 
            #       preds.cpu().shape, fea_ori.cpu().shape)

    avg_loss = test_loss / len(test_loader)
    acc = test_correct / test_total
    all_sam_md = torch.cat(all_sam_md).view(-1)
    all_preds = torch.cat(all_preds).view(-1)
    all_targets = torch.cat(all_targets).view(-1)
    all_fea_ori = torch.cat(all_fea_ori, dim=0) #默认dim=0
    all_fea_ori = all_fea_ori.flatten(start_dim=0, end_dim=1)  ##或者view(-1, all_fea_ori.shape[-1])也一样 reshape也是如此
    per_class_acc = compute_per_class_accuracy(all_preds, all_targets, num_class)

    # ===== 新增 F1-score =====
    y_true = all_targets.view(-1).numpy()
    y_pred = all_preds.view(-1).numpy()
    f1_macro = f1_score(y_true, y_pred, average="macro")  # 整体 F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, 
                            labels=list(range(num_class)), zero_division=0)  # 每类 F1
    print("可能岩性类别并未出现全，这是计算F1-score的labels参数设置全可能并不好")

    return avg_loss, acc, per_class_acc, all_preds, all_targets, all_fea_ori, f1_macro, per_class_f1, all_sam_md



def process_wells(dfTest, args):
    """
    功能：
        - 获取所有井名，并将其中的 '/' 替换为 '-'
        - 统计每口井的样本个数
    参数：
        dfTest : DataFrame，包含 'WELL' 列
        args  : dict，包含键 'sam_len'，表示样本长度
    
    返回：
        all_well_name    : 原始井名数组
        all_well_name_r  : 替换后的井名数组
        all_well_len     : 每口井的样本点数（Series）
        num_siw          : 每口井中能分割出的样本个数（np.ndarray）
    """
    # 原始井名
    all_well_name = dfTest[args.wl[0]].unique()

    # 替换掉井名中的 '/'
    all_well_name_r = [w.replace('/', '-') for w in all_well_name]

    # 每口井的长度
    all_well_len = dfTest.groupby(args.wl[0], sort=False).size()

    # 每口井可分割的样本个数
    num_siw = np.array(all_well_len / args.sam_len, dtype=int)

    # 打印信息
    # print('all_well_name:', all_well_name)
    # print('all_well_name_r:', all_well_name_r)
    # #print('all_well_len:', all_well_len)
    # print('每口井中样本的个数:', num_siw)

    return all_well_name, all_well_name_r, num_siw




# ===== 新增：工具函数 =====
def compute_overall_metrics(all_preds, all_targets, num_classes):
    """计算整体 Accuracy、F1、
    Per-class Accuracy, Per-class F1, 
    Confusion Matrix（数量 & 比例）"""

    all_preds = all_preds.view(-1).numpy()
    all_targets = all_targets.view(-1).numpy()

    # Overall Accuracy
    overall_acc = (all_preds == all_targets).mean()

    # F1
    overall_f1 = f1_score(all_targets, all_preds, average='macro') #weighted

    # Per-class F1
    per_class_f1 = f1_score(
        all_targets, all_preds,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0
    )

    # Per-class Accuracy
    per_class_acc = []
    for cls in range(num_classes):
        true_mask = (all_targets == cls)
        if true_mask.sum() == 0:
            per_class_acc.append(np.nan)
        else:
            per_class_acc.append((all_preds[true_mask] == cls).mean())

    # Confusion Matrix  #该矩阵是计数矩阵，
    cm = confusion_matrix(
        all_targets, all_preds,
        labels=list(range(num_classes))
    )
    # ===== 新增：比例混淆矩阵（按行归一化）=====
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    return overall_acc, overall_f1, per_class_acc, per_class_f1, cm_norm




# ========== 主程序开始 ==========
if __name__ == "__main__":
    args = read_args()
    device = torch.device('cpu')

    dftrain, dfval, dftest = get_dataset(args)

    dftrain_spec = np.load(args.train_set_spec)
    dfval_spec = np.load(args.val_set_spec)
    dftest_spec = np.load(args.test_set_spec)
    print("======== dftrain_spec.shape, dfval_spec.shape, dftest_spec.shape \n",
        dftrain_spec.shape, dfval_spec.shape, dftest_spec.shape)
    
    # print("======= 使用 dftrain 做测试集")
    # dftest = dftrain  #.iloc[:100000, :]
    # dftest_spec = dftrain_spec  #[:100000, :]

    # print("======= 使用 dfval 做测试集")
    # dftest = dfval
    # dftest_spec = dfval_spec

    print("======= 使用 dftest 做测试集")
    print("====== dftest.shape, 井数目:", dftest.shape, len(dftest[args.wl[0]].unique()))

    # 岩性映射
    if len(args.lith_code_map_name.keys()) == 4:
        dftest = lith_code_remap_int_exclude_few_lith(dftest, args.lith[0], args.lith_code_map_name)
    elif len(args.lith_code_map_name.keys()) == 6:
        dftest = lith_code_remap_int_exclude_few_lith_6classes(dftest, args.lith[0], args.lith_code_map_name)
    elif len(args.lith_code_map_name.keys()) == 5 and args.data_area.__contains__("yuanba"):
        dftest = lith_code_remap_int_yuanba_5class(dftest, args.lith[0], args.lith_code_map_name)
    else:
        dftest = lith_code_remap_int(dftest, args.lith[0], args.lith_code_map_name)
    
    # # mask
    dftest_well_maskc, dftest_loc_li = well_location_mask(dftest, args.sam_len, args)

    criterion = torch.nn.CrossEntropyLoss()

    if args.stage in ["pretrain", "finetune"]:
        model = LogGPT_Chunk16(args).float().to(device)
    elif args.stage == "diretrain":  # 用于各种 baseline 模型的训练
        # ====== 根据参数选择模型 ======
        if args.model_type == "LogGPT_Chunk16":
            print("====== 使用 LogGPT_Chunk16（无预训练）")
            model = LogGPT_Chunk16(args).to(device)
        elif args.model_type == "LogTransformer":
            print("====== 使用 Transformer")
            model = LogTransformer(args).to(device)
        elif args.model_type == "LogResNet":
            print("====== 使用 ResNet")
            model = LogResNet(args).to(device)
        elif args.model_type == "CNNBiLSTM":
            print("====== 使用 CNNBiLSTM")
            model = CNNBiLSTM(args).to(device)
        else:
            raise ValueError(f"====== ⚠️⚠️⚠️ Unsupported model: {args.model_type}")





    experiment_name = args.experiment_name
    print(f"====== experiment_name:{experiment_name}")
    # Load model
    model_name = "2.7_best_model_exp57_epo5903_0p3943.pth"
    #"1.21_best_model_exp53_epo7959_0p4784.pth" #"best_model.pth"  #"11.7_best_model_exp12_3144epo.pth"  #'11.29_best_model_exp19_3985epo.pth'
    #model_name = "1.8_best_model_exp35_645epo.pth"
    print(f"====== 模型名称：{model_name}")

    set_fig_id = 'test_set_'   ##保存的前缀，用于著名是测试集还是其他
    save_path = f'../fig/feas_preds_labels/test_fig/{experiment_name}/'  ## train_fig  test_fig  val_fig
    os.makedirs(save_path, exist_ok=True)
    args.mode = 'test'   ##用于data_loader中构建样本的方式
    print(f"====== 预测图片保存位置:{save_path}")




    ## 载入最佳模型权重
    best_model_path = os.path.join(f'../model/{experiment_name}/', model_name)
    checkpoint = torch.load(best_model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        # model.load_state_dict(checkpoint["model_state_dict"])
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"])
        print("====== Missing:", missing)
        print("====== Unexpected:", unexpected)
    else:
        model.load_state_dict(checkpoint)
        print("====== 错误: Model loaded without 'model_state_dict' key.")
    print(f"====== Loaded best model from {best_model_path}")

    all_well_name, all_well_name_r, num_siw = process_wells(dftest, args)

    # ====== 全局收集 ======
    all_preds_list = []
    all_targets_list = []
    all_sam_mds_list = []

    well_acc = []
    well_f1 = []
    well_names = []

    # 类别标签用于画图
    class_ids = list(range(len(args.lith_code_map_name)))
    class_labels = list(args.lith_code_map_name.values())

    print("====dftest_loc_li", dftest_loc_li)
    for id in range(len(all_well_name)):
        wname = all_well_name[id]
        print(f"====== {wname}井效果")
        siw = int(num_siw[id])

        well = dftest[dftest.WELL == wname].copy()
        
        test_dataset = BertDataset_by_sam(well, dftest_spec, args.sam_len, siw,
                                          dftest_well_maskc, dftest_loc_li, args)
        test_loader = DataLoader(test_dataset, batch_size=siw, shuffle=False, num_workers=12)

        # Run test
        test_loss, test_acc, test_per_class_acc, preds, targets, fea_oris, f1_macro, per_class_f1, sam_mds = \
            test_one_epoch(model, test_loader, criterion, device, num_class=len(args.lith_code_map_name.keys()))
        print('preds.shape, targets.shape, fea_oris.shape', preds.shape, targets.shape, fea_oris.shape)

        ### 收集所有井数据
        all_preds_list.append(preds)
        all_targets_list.append(targets)
        all_sam_mds_list.append(sam_mds) #sam_mds是一口井的所有样本的sam_md拼接而成的

        ### 收集井指标
        well_acc.append(test_acc)
        well_f1.append(f1_macro)
        well_names.append(all_well_name_r[id])

        ##可视化预测结果
        visualize_logs_preds_targets(fea_oris, preds, targets, 
                                     args.lith_code_map_name,
                                 save=save_path, wname=all_well_name_r[id])
        
        ##保存结果
        # ===== 拼接张量 =====
        # 将 preds 和 targets 扩展为二维再拼接
        pt = torch.cat([sam_mds.unsqueeze(1), fea_oris, preds.unsqueeze(1), targets.unsqueeze(1)], dim=1)                   
        # ===== 转换为 pandas DataFrame =====
        cs = ['DEPTH_MD'] + [fea for fea in args.fea] + ['Pred', 'Target']
        ptdf = pd.DataFrame(pt.cpu().numpy(), columns=cs)
        # ===== 添加指标列，只首行填值 =====
        ptdf['loss'] = ''
        ptdf['accuracy'] = ''
        ptdf['F1'] = ''
        ptdf["per_class_acc/Recall"] = ''
        ptdf["per_class_f1"] = ''
        ptdf.loc[0, 'loss'] = f"{test_loss:.4f}"
        ptdf.loc[0, 'accuracy'] = f"{test_acc:.4f}"
        ptdf.loc[0, 'F1'] = f"{f1_macro:.4f}"
        # 这个数据应该与下述 per_class_acc_heatmap.jpg 图中一致
        ptdf.loc[0, 'per_class_acc/Recall'] = ', '.join([f"{acc:.4f}" for acc in test_per_class_acc])
        ##这个数据应该与下述 per_class_f1_heatmap.jpg 图中一致
        ptdf.loc[0, 'per_class_f1'] = ', '.join([f"{f1:.4f}" for f1 in per_class_f1])
        # ===== 保存为 CSV 文件 =====
        ptdf.to_csv(save_path + all_well_name_r[id]+'.csv', index=False, encoding='utf-8-sig')

        print("======= Test Results =======")
        print(f"Test Loss of well {wname}: {test_loss:.4f}")
        print(f"Test Accuracy of well {wname}: {test_acc:.4f}")
        print(f"Macro F1-score of well {wname}: {f1_macro:.4f}")   # 单口井的F1

        ###各类的acc F1情况
        # for c, acc in enumerate(test_per_class_acc):
        #     print(f"Class {c} Accuracy: {acc:.4f} | F1-score: {per_class_f1[c]:.4f}")



    # ===== 拼接整体结果 =====
    all_preds = torch.cat(all_preds_list).view(-1)
    all_targets = torch.cat(all_targets_list).view(-1)

    overall_acc, overall_f1, overall_per_class_acc, overall_per_class_f1, cm = \
        compute_overall_metrics(all_preds, all_targets, len(class_ids))

    print("\n================ 全部井总体评估结果 ================")
    print(f"Overall Accuracy = {overall_acc:.4f}")
    print(f"Overall F1 = {overall_f1:.4f}")
    print("Per-class Accuracy:")
    print(overall_per_class_acc)
    print("Per-class F1:")
    print(overall_per_class_f1)
    # ===== 保存整体指标(整体的Acc、F1, 每类的Acc、F1)=====
    data = {
        "Overall Accuracy": [overall_acc],
        "Overall F1": [overall_f1]
    }
    # 添加 per-class 指标（按类名作为列）
    for cls_name, acc, f1 in zip(class_labels, overall_per_class_acc, overall_per_class_f1):
        data[f"{cls_name} Acc"] = [acc]
        data[f"{cls_name} F1"] = [f1]

    df_metrics = pd.DataFrame(data)
    df_metrics.to_csv(save_path + set_fig_id + "overall_metrics.csv", encoding="utf-8-sig", index=False)

    # ===== 保存混淆矩阵 =====
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.to_csv(save_path + set_fig_id + "confusion_matrix.csv", encoding="utf-8-sig")
    ###可视化混淆矩阵
    save_confusion_matrix_heatmap(cm, class_labels, save_path, name=set_fig_id + "confusion_matrix")

    # ===== 绘制 Accuracy & F1 柱状图 =====
    plot_bar(well_acc, labels=well_names, ylabel="Accuracy", 
             title="Well Accuracy", 
             save_path=save_path + set_fig_id + "well_accuracy_bar.jpg")
    plot_bar(well_f1, labels=well_names, ylabel="F1-score",
              title="Well F1-score", 
              save_path=save_path + set_fig_id + "well_f1_bar.jpg")
    


    # ============= 每类别 Accuracy(即是召回率)（同compute_per_class_accuracy中的per_class_acc结果相同） =============
    per_well_per_class_acc = []

    for p, t in zip(all_preds_list, all_targets_list):
        acc_i = []

        y_pred = p.numpy()
        y_true = t.numpy()

        for cls in class_ids:
            mask = (y_true == cls)
            if mask.sum() == 0:
                acc_i.append(np.nan)
            else:
                acc_i.append((y_pred[mask] == cls).mean())

        per_well_per_class_acc.append(acc_i)

    acc_matrix = np.array(per_well_per_class_acc)

    plot_heatmap(acc_matrix, well_names, class_labels, "Cross well-category Accuracy heatmap",
                 save_path + set_fig_id + "per_class_accuracy_heatmap.jpg")



