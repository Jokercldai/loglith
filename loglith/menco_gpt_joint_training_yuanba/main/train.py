
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
# 确认设备映射
print("Visible devices:", os.environ["CUDA_VISIBLE_DEVICES"]) 
print("Actual device:", torch.cuda.current_device()) 
print("Visible device count: torch.cuda.device_count()", torch.cuda.device_count())
print("Device 0 name: torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
print("================================================")



import torch
import numpy as np
import time
import pandas as pd
import sys

from config import read_args
from data_loader import *
from model import * 
from model import * 
from model_bert import * 
from model_contrast import *

from utils import *

from utils_contrast import contrast_train_network

from model_bert import LogBERT
from utils_bert import pretrain_network_bert, train_network_bert

sys.path.append("/home/cldai/sciresearch/loglith")
from toolkit.plot_fig import *


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def train_model(args):


    print('====== args.data_dir:', args.data_dir)
    print('====== args.sam_len:', args.sam_len)
    
    print("====== 样本重叠尺寸:", args.overlap_len)
    print('====== args.sam_num:', args.sam_num)
    print('====== args.bsize:', args.bsize)
    print("====== args.chunk_size", args.chunk_size)
    print('====== args.load_mode:', args.load_mode)
    print("====== args.lr:", args.lr)
    print('====== args.epochs:', args.epochs)
    print("====== 预训练 / 微调 ?:", args.stage)

    print("====== 训练集:", args.train_set)
    print("====== 验证集:", args.val_set)
    print("====== 测试集:", args.test_set)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ##读取需要的列数据, 如['WELL', 'DEPTH_MD', 'GR','RHOB', 'NPHI', 'DTC', 'RDEP', 'LITH']
    dftrain, dfval, dftest = get_dataset(args)
    print("======= 数据读取完成！！！", dftrain.shape, dfval.shape, dftest.shape)

    ##确认岩性数据是完整的  避免不完整造成样本制作和后续预测的错误
    miss_a = dftrain[args.lith].isnull().sum()
    miss_b = dfval[args.lith].isnull().sum()
    miss_c = dftest[args.lith].isnull().sum()
    ##下述输出显示所有井的岩性数据是完整的
    print("======= 岩性数据缺失的数目:miss_train, miss_val, miss_test: \n", 
          miss_a, '\n', miss_b, '\n', miss_c)

    ##岩性数据映射(原始的岩性编码映射至新的)  lith_code_remap_01_range
    ## args.lith[0]取[0]是为了dftrain[args.lith[0]]是Series，这样可以映射  
    ##lith_code_remap_int 函数将岩性应设置[0, 岩性类别-1]
    if len(args.lith_code_map_name.keys()) == 4:
        dftrain = lith_code_remap_int_exclude_few_lith(dftrain, args.lith[0], args.lith_code_map_name)
        dfval = lith_code_remap_int_exclude_few_lith(dfval, args.lith[0], args.lith_code_map_name)
        dftest = lith_code_remap_int_exclude_few_lith(dftest, args.lith[0], args.lith_code_map_name)
    elif len(args.lith_code_map_name.keys()) == 6:
        dftrain = lith_code_remap_int_exclude_few_lith_6classes(dftrain, args.lith[0], args.lith_code_map_name)
        dfval = lith_code_remap_int_exclude_few_lith_6classes(dfval, args.lith[0], args.lith_code_map_name)
        dftest = lith_code_remap_int_exclude_few_lith_6classes(dftest, args.lith[0], args.lith_code_map_name)
    elif len(args.lith_code_map_name.keys()) == 5 and args.data_area.__contains__("yuanba"):
        dftrain = lith_code_remap_int_yuanba_5class(dftrain, args.lith[0], args.lith_code_map_name)
        dfval = lith_code_remap_int_yuanba_5class(dfval, args.lith[0], args.lith_code_map_name)
        dftest = lith_code_remap_int_yuanba_5class(dftest, args.lith[0], args.lith_code_map_name)
    else:
        dftrain = lith_code_remap_int(dftrain, args.lith[0], args.lith_code_map_name)
        dfval = lith_code_remap_int(dfval, args.lith[0], args.lith_code_map_name)
        dftest = lith_code_remap_int(dftest, args.lith[0], args.lith_code_map_name)
    # print("====== 岩性编码重映射后 dftrain.head():\n", dftrain.head())
    # print("====== 岩性编码重映射后 dfval.head():\n", dfval.head())
    # print("====== 岩性编码重映射后 dftest.head():\n", dftest.head())
    ##临时保存一下映射后的数据，观察是否映射正确
    #dftest.to_csv("/home/cldai/sciresearch/loglith/data/linshi/"+'dftest_new_code_map.csv', index=False)

    # dftrain_spec = np.load(args.train_set_spec)
    # dfval_spec = np.load(args.val_set_spec)
    # dftest_spec = np.load(args.test_set_spec)
    dftrain_spec = dftrain
    dfval_spec = dfval
    dftest_spec = dftest
    print("======== dftrain_spec.shape, dfval_spec.shape, dftest_spec.shape \n",
          dftrain_spec.shape, dfval_spec.shape, dftest_spec.shape)

    ### 井位置mask (在两井连接处设置mask，以避免跨井制作样本)
    dftrain_well_maskc, dftrain_loc_li = well_location_mask(dftrain, args.sam_len, args)
    dfval_well_maskc, dfval_loc_li = well_location_mask(dfval, args.sam_len, args)
    dftest_well_maskc, dftest_loc_li = well_location_mask(dftest, args.sam_len, args)
    ##可视化井拼接处的mask   #plot_len:截取一定长度的数据来绘制maskc
    # visualize_maskc(dftrain_well_maskc, dftrain, "train", 
    #                 plot_len=80000, save_path=args.well_loc_mask)
    # visualize_maskc(dfval_well_maskc, dfval, "val", 
    #                 plot_len=80000, save_path=args.well_loc_mask)
    # visualize_maskc(dftest_well_maskc, dftest, "test", 
    #                 plot_len=80000, save_path=args.well_loc_mask)

    train_sam_num = int((len(dftrain)-args.sam_len)/(args.overlap_len))   
    val_sam_num = int((len(dfval)-args.sam_len)/(args.overlap_len))
    print("====== Dataset_regression中使用train_sam_num / val_sam_num代替固定的args.sam_num")
    print(f"====== train_sam_num:{train_sam_num}, val_sam_num:{val_sam_num}")
    
    ####加载数据
    train_dataset = BertDataset_by_sam(dftrain, dftrain_spec, args.sam_len, 
                                       train_sam_num, 
                                       #args.sam_num, 
                                dftrain_well_maskc, dftrain_loc_li, args)
    val_dataset = BertDataset_by_sam(dfval, dfval_spec, args.sam_len, 
                                     val_sam_num, 
                                     #args.sam_num, 
                                dfval_well_maskc, dfval_loc_li, args)
    train_loader = DataLoader(dataset=train_dataset, batch_size = args.bsize, 
						   shuffle=True, num_workers=args.num_workers,
               pin_memory=True,            #加速GPU数据传输
               persistent_workers=True,    #减少进程创建开销
               worker_init_fn=worker_init_fn,
               )
    val_loader = DataLoader(dataset=val_dataset, batch_size = args.bsize, 
						   shuffle=True, num_workers=args.num_workers,
                           worker_init_fn=worker_init_fn)
    

    if args.stage == "pretrain":
        print("====== 预训练阶段！！！")

        if args.model_type in ["LogGPT_Chunk16", "LogGPT_Chunk32", "LogGPT_Chunk1"]:
            print(f"====== 使用 {args.model_type} 模型进行预训练")
            model = LogGPT_Chunk16(args).to(device)
            #print("======== 查看模型结构: \n", model, '\n========================') 
            pretrain_network(args, model, train_loader, val_loader, device, )
        elif args.model_type == "LogBERT":
            print("====== 使用 LogBERT 模型进行预训练")
            model = LogBERT(args).to(device)
            #print("======== 查看模型结构: \n", model, '\n========================') 
            pretrain_network_bert(args, model, train_loader, val_loader, device, )
        else:
            raise ValueError(f"====== ⚠️⚠️⚠️ Unsupported model for pretraining: {args.model_type}")
        


    elif args.stage == "finetune":
        print("====== 微调阶段！！！")

        pretrain_path = '../model/' + "pretrain_" + args.model_type + "/" +args.pretrained_model
        args.resume_ckpt = pretrain_path

        if args.model_type in ["LogGPT_Chunk16", "LogGPT_Chunk32", "LogGPT_Chunk1"]:
            print(f"====== 使用 {args.model_type} 模型进行微调")
            model = LogGPT_Chunk16(args).to(device)

            train_network(args, model, train_loader, val_loader, 
                      num_class = len(args.lith_code_map_name.keys()), 
                      device=device, freeze_encoder=True)

        elif args.model_type in ["LogBERT", "LogBERT_parttune"]:
            print("====== 使用 LogBERT 模型进行微调")
            model = LogBERT(args).to(device)    

            train_network_bert(args, model, train_loader, val_loader, 
                      num_class = len(args.lith_code_map_name.keys()), 
                      device=device, freeze_encoder=True)
        else:
            raise ValueError(f"====== ⚠️⚠️⚠️ Unsupported model for finetune: {args.model_type}")
        

    
    elif args.stage == "diretrain":  # 用于各种 baseline 模型的训练
        print("====== 直接训练模式（无预训练）！！！网络模型消融/对比实验！！！")

        # ====== 根据参数选择模型 ======
        if args.model_type in ["LogGPT_Chunk16", "LogGPT_Chunk32", "LogGPT_Chunk1"]:
            print(f"====== 使用 {args.model_type}（无预训练）, 注意此时")
            model = LogGPT_Chunk16(args).to(device)

            train_network(args, model, train_loader, val_loader, 
                      num_class = len(args.lith_code_map_name.keys()), 
                      device=device, freeze_encoder=True)
        else:
            if args.model_type == "LogTransformer":
                print("====== 使用 LogTransformer")
                model = LogTransformer(args).to(device)

            elif args.model_type == "LogResNet":
                print("====== 使用 LogResNet")
                model = LogResNet(args).to(device)

            elif args.model_type == "CNNBiLSTM":
                print("====== 使用 CNNBiLSTM")
                model = CNNBiLSTM(args).to(device)

            else:
                raise ValueError(f"====== ⚠️⚠️⚠️ Unsupported model: {args.model_type}")

            # # 开始训练
            contrast_train_network(args, model, train_loader, val_loader, 
                        num_class = len(args.lith_code_map_name.keys()), 
                        device=device, freeze_encoder=False)



def worker_init_fn(worker_id):
    # 每个 worker 的随机种子不同
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args = read_args()
    train_model(args)