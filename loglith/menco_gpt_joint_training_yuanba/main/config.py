import argparse
import os


def read_args():
    parser = argparse.ArgumentParser()
    
    ################
    ##### fig_path #####
    ###############
    parser.add_argument("--well_loc_mask", type=str, 
                        default="/home/cldai/sciresearch/loglith/figure/well_loc_mask/", 
                        help="可视化井数据之间拼接处的mask")
    parser.add_argument("--loss_fig", type=str, 
                        default="/home/cldai/sciresearch/loglith/menco/fig/loss/", help="loss图存放位置") 
    

    # The dataset  
    parser.add_argument(
        "--data_dir",
        type=str,     #cldai
        default="/home/cldai/sciresearch/loglith/data/yuanba/processed/")
    parser.add_argument("--train_set", type=str, default="train.csv",  
                        help="训练集") 
    
    parser.add_argument("--val_set", type=str, default="val_test.csv",  
                        help="验证集")
    
    ## 集合验证集和测试集  
    parser.add_argument("--test_set", type=str, default="val_test.csv",   
                        help="测试集")
    
    parser.add_argument(
        "--spec_data_dir",
        type=str,     #cldai
        default="/home/cldai/sciresearch/loglith/data/AIgeo_for_lp/")
    parser.add_argument("--train_set_spec", type=str, default="train_set_spec.npy", help="训练集")
    parser.add_argument("--val_set_spec", type=str, default="val_set_spec.npy", help="验证集")
    parser.add_argument("--test_set_spec", type=str, default="test_set_spec.npy", help="测试集")


    ### 关于列名
    parser.add_argument("--befroe_rename_col", type=list, help="数据的col改名，原始col名",
                        default=['LITH'])
    parser.add_argument("--after_rename_col", type=list, help="数据的col改名，改后col名",
                        default=['LITH'])
    parser.add_argument("--well_md_fea_lith", type=list, help="需要的数据的col名",
                        default=['WELL', 'DEPTH_MD', 'GR', 'DEN', 'CNL', 'AC', 
                                  'RD',  'LITH'])  #'RS', 'KTH',
    parser.add_argument("--md", type=list, help="深度的列名",
                        default=['DEPTH_MD'])
    parser.add_argument("--wl", type=list, help="井名的列名",
                        default=['WELL'])
    parser.add_argument("--fea_lith", type=list, help="曲线和岩性名称列表",
                        default=['GR', 'DEN', 'CNL', 'AC', 
                                  'RD', 'LITH'])
    parser.add_argument("--fea", type=list, help="曲线列表",
                        default=['GR', 'DEN', 'CNL', 'AC', 'RD'])
    parser.add_argument("--lith", type=list, help="曲线列表",
                        default=['LITH'])
    
    ## 测井曲线时频分析频谱范围
    parser.add_argument("--log_freq_range", type=list, help="测井曲线时频分析频谱范围",
                        default=[20, 84])
    
    # ##原始的岩性数据
    parser.add_argument("--lith_code_map_name", type=dict, help="岩性编码和名称的映射",
                        default={

                                0: "Mudstone",
                                1: "Sandy Mudstone",
                                2: "Muddy Sandstone",
                                3: "Sandstone",
                                4: "Conglomeratic Sandstone",
                                #5: "Carbonaceous Mudstone",
                                #6: "Coal",
                                #7: "Carbonaceous Shale",
                                }
                        )
                                # 0: "Mudstone",
                                # 1: "Sandy_Mud",
                                # 2: "Muddy_Sand",
                                # 3: "Sandstone",
                                # 4: "Conglomeratic_Sand",
                                # #5: "Carbonaceous_Mud",
                                # #6: "Coal",
                                # #7: "Carbonaceous_Shale",
                                # }
    


    ## 针对不同数据集需要更改
    parser.add_argument("--lith_weight", type=list, help="各岩性数量，可作为loss函数的权重设计",
                        default=[1.00, 1.2, 1.5, 3.00, 3.5,]
                        ) 
                        ## default=[1.00, 1.01, 1.08, 1.53, 2.09] 对比的网络用
                        ## default=[1.00, 1.01, 1.08, 1.53, 2.09, 9.23, 25.11, 35.51]
                        #岩性数目比例：[31521, 30622, 26953, 13397, 7201, 370, 50, 25]
    # parser.add_argument("--lith_weight", type=list, help="各岩性数量，可作为loss函数的权重设计",
    #                     default=[0.328, 0.157, 0.352, 0.419, 3.513, 1.231]
    #                     )


    

    ####################### 下述是为了CNN-LSTM模型设计的参数。  ########################
    
    # parser.add_argument("--sam_len", default=512, type=int, help="构建的样本的长度")
    # parser.add_argument("--overlap_len", default=35, type=int, help="样本重叠长度")
    # parser.add_argument("--sam_num", default=5, type=int, help="每轮构建样本数量")
    # parser.add_argument("--bsize", default=300, help="batch size")      ##original is 64  之前微调等用的是200bsize
    # parser.add_argument("--lr", default=0.00002, help="learning rate") 
    # parser.add_argument("--num_workers", default=8, help="number of workers for data loader") 
    # parser.add_argument("--chunk_size", default=1, help="通过chunk实现连续曲线值的token化") 
    # parser.add_argument("--grad_clip", default=1.0, help="梯度裁剪的阈值") 
    # parser.add_argument("--data_area", default="yuanba", type=str, help="数据区域名称，用于岩性编码映射") 

    # parser.add_argument("--epochs", default=3, help="number of epochs")  ##ori  1000
    
    

    # ## data_loader.py中使用，用于控制训练/测试阶段样本的生成方式
    # parser.add_argument("--load_mode", default='test', type=str, 
    #                     help="训练 train or 测试 test 阶段(需调整),用于加载数据")
    
    # ### 主程序中使用，用于控制 训练/微调 阶段，用于控制加载的预训练模型
    # parser.add_argument("--stage", type=str, default="diretrain",   ## finetune
    #                     help="训练阶段： pretrain or finetune or diretrain")   ## pretrain
    
    # parser.add_argument("--model_type", type=str, default="LogTransformer",   
    #                     help=""" 使用的模型框架：
    #                     pretrain or finetune阶段不用设置该参数也可，默认使用 LogGPT,但为了后续画图和保存方便，最好加上
    #                     diretrain 阶段使用LogGPT时需要使用 LogGPT 参数
    #                     框架： LogGPT  LogGPT_Chunk  LogGPT_Chunk16  LogTransformer  
    #                     CNNBiLSTM  LogResNet  
    #                     """)
    
    # parser.add_argument("--loss_type", type=str, default="focal_tversky",
    #                     choices=["ce", "focal", "balanced_focal", "focal_tversky"],
    #                     help="""选择损失函数类型：ce = CrossEntropy | focal = FocalLoss | 
    #                             balanced_focal = 带类别权重的Focal | 
    #                             focal_tversky = 针对薄层和稀有类"""
    #                 )
    




    # # # # # # # ######################## 下述是为了  LogGPT /  LogBERT 512设计的参数  ########################
    parser.add_argument("--sam_len", default=512, type=int, help="构建的样本的长度")
    parser.add_argument("--overlap_len", default=35, type=int, help="样本重叠长度")
    parser.add_argument("--sam_num", default=5, type=int, help="每轮构建样本数量")
    parser.add_argument("--bsize", default=300, help="batch size")      ##original is 64  之前微调等用的是200bsize
    parser.add_argument("--lr", default=0.00002, help="learning rate") 
    parser.add_argument("--main_net_lr", default=0.00002, help="微调时主网络的学习率") 
    parser.add_argument("--num_workers", default=8, help="number of workers for data loader") 
    parser.add_argument("--grad_clip", default=1.0, help="梯度裁剪的阈值") 
    parser.add_argument("--data_area", default="yuanba", type=str, help="数据区域名称，用于岩性编码映射") 

    parser.add_argument("--epochs", default=3, help="number of epochs")  ##ori  1000
    parser.add_argument("--chunk_size", default=8, help="通过chunk实现连续曲线值的token化") 



    ## data_loader.py中使用，用于控制训练/测试阶段样本的生成方式
    parser.add_argument("--load_mode", default='test', type=str, 
                        help="训练 train or 测试 test 阶段(需调整),用于加载数据")
    
    ### 主程序中使用，用于控制 训练/微调 阶段，用于控制加载的预训练模型
    parser.add_argument("--stage", type=str, default="diretrain",   ## finetune
                        help="训练阶段： pretrain or finetune or diretrain")   ## pretrain
    
    parser.add_argument("--model_type", type=str, default="LogGPT_Chunk16",   
                        help=""" 使用的模型框架：
                        pretrain or finetune阶段不用设置该参数也可，默认使用 LogGPT,但为了后续画图和保存方便，最好加上
                        diretrain 阶段使用LogGPT时需要使用 LogGPT 参数
                        框架： LogGPT  LogGPT_Chunk  LogGPT_Chunk16  
                        LogGPT_Chunk32  LogGPT_Chunk1
                        LogTransformer  CNNBiLSTM  LogResNet  
                        LogBERT LogBERT_parttune
                        LogGPT_Chunk512_linshi
                        """)
    
    ### 微调训练时加载的预训练模型
    parser.add_argument("--pretrained_model", type=str, 
                        default= ".pth", #"1.21_best_pretrain_model_exp50_epo3759_0p0041.pth", 
                        #"",  ##  1.13_epoch_3500_exp40.pth  finetune。best_pretrain_model
                        help="加载的预训练模型")

    parser.add_argument("--loss_type", type=str, default="focal_tversky",
                        choices=["ce", "focal", "balanced_focal", "focal_tversky"],
                        help="""选择损失函数类型：ce = CrossEntropy | focal = FocalLoss | 
                                balanced_focal = 带类别权重的Focal | 
                                focal_tversky = 针对薄层和稀有类"""
                    )




    args = parser.parse_args()

    ## 通过后处理将数据文件名何其路径拼接起来
    args.train_set = os.path.join(args.data_dir, args.train_set)
    args.val_set   = os.path.join(args.data_dir, args.val_set)
    args.test_set  = os.path.join(args.data_dir, args.test_set)

    args.train_set_spec = os.path.join(args.spec_data_dir, args.train_set_spec)
    args.val_set_spec   = os.path.join(args.spec_data_dir, args.val_set_spec)
    args.test_set_spec  = os.path.join(args.spec_data_dir, args.test_set_spec)

    args.experiment_name = f"{args.stage}_{args.model_type}"

    return args
 