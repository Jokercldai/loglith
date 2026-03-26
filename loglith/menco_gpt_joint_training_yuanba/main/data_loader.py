import pandas as pd
import numpy as np
import os
import torch
import random 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from scipy.special import softmax
import sys 
from torch.utils.data import Dataset, DataLoader
import pywt
import math




def get_dataset(args):
    dftrain=pd.read_csv(args.train_set,engine='python')
    dfval=pd.read_csv(args.val_set,engine='python')
    dftest=pd.read_csv(args.test_set,engine='python')

    # === 1. 执行列重命名 ===
    rename_dict = dict(zip(args.befroe_rename_col, args.after_rename_col))
    print("===== 原始df数据中需要改的列名:", rename_dict)
    dftrain.rename(columns=rename_dict, inplace=True)
    dfval.rename(columns=rename_dict, inplace=True)
    dftest.rename(columns=rename_dict, inplace=True)

    # === 2. 筛选指定列 ===
    cols = args.well_md_fea_lith
    dftrain = dftrain[cols].copy()
    dfval = dfval[cols].copy()
    dftest = dftest[cols].copy()
    # print("====== dftrain.head():\n", dftrain.head())
    # print("====== dfval.head():\n", dfval.head())
    # print("====== dftest.head():\n", dftest.head())

    return dftrain, dfval, dftest


def lith_code_remap_01_range(dfdata, lith_col, lith_code_map_name):
    print("====== 将岩性编码重新映射到[0,1]范围")

    lith_code = list(lith_code_map_name.keys())
    min_code = min(lith_code)
    max_code = max(lith_code)
    dfdata[lith_col] = (dfdata[lith_col]- min_code) / (max_code - min_code)

    return dfdata


def lith_code_remap_01_range(dfdata, lith_col, lith_code_map_name):
    print("====== 将岩性编码重新映射到[0,num_class]范围内的整数")

    lith_code = list(lith_code_map_name.keys())
    min_code = min(lith_code)
    max_code = max(lith_code)
    dfdata[lith_col] = (dfdata[lith_col]- min_code) / (max_code - min_code)

    return dfdata

def lith_code_remap_int(dfdata, lith_col, lith_code_map_name):
    print("====== 将岩性编码重新映射到 [0, 类别数-1] 整数范围")

    # 获取岩性代码并排序
    lith_codes = sorted(lith_code_map_name.keys())

    # 构建映射字典 {原始编码: 新的整数编码}
    code2int = {code: idx for idx, code in enumerate(lith_codes)}
    # 按照映射关系替换原始数据
    dfdata[lith_col] = dfdata[lith_col].map(code2int)

    print("====== 岩性代码重新映射关系 code2int: \n", code2int)
    
    return dfdata

def lith_code_remap_int_exclude_few_lith(dfdata, lith_col, lith_code_map_name):
    """
    将岩性编码重新映射到整数范围：
      30000 -> 0
      65000 -> 1
      65030 -> 2
      其他 (包括 lith_code_map_name 中除这三个外的所有 key 以及 dfdata 中出现的未知编码) -> 3
    """

    print("====== 将岩性编码重新映射到 [0, 3] 范围(四分类)（前3类为指定岩性，其余岩性统一为3!!!）")
    
    # 需要单独映射的三类
    main_codes = [30000, 65000, 65030]

    # 构建初始映射字典
    code2int = {30000: 0, 65000: 1, 65030: 2}
    print(f"====== 岩性编码映射字典:{code2int}; 注意：若岩性不同，则需要更改映射字典")

    # 收集所有 dfdata 中的编码
    unique_codes_in_data = dfdata[lith_col].unique().tolist()

    # 找出 dfdata 中的“其他类”编码
    other_codes = [code for code in unique_codes_in_data if code not in main_codes]

    # 把所有“其他类”编码映射为 3
    for code in other_codes:
        code2int[code] = 3

    # 应用映射
    dfdata[lith_col] = dfdata[lith_col].map(code2int).fillna(3).astype(int)

    # # 输出映射关系
    # print("====== 岩性代码重新映射关系 code2int:")
    # for k, v in sorted(code2int.items()):
    #     print(f"  {k} -> {v} ({lith_code_map_name.get(k, 'Unknown/Unlisted')})")

    return dfdata


def lith_code_remap_int_exclude_few_lith_6classes(dfdata, lith_col, lith_code_map_name):
    """
    将岩性编码重新映射到整数范围：
      30000 -> 0
      65000 -> 1 ##90000: 'Coal',  93000: 'Basement', 99000: 'Tuff',
      65030 -> 2
      70000 -> 3 #70032: 'Chalk',  74000: 'Dolomite',  80000: 'Marl',
      86000 -> 4
      88000 -> 5
    """

    print("====== 将岩性编码重新映射到 [0, 5] 范围(六分类)!!!")
    
    # 构建初始映射字典
    code2int = {30000: 0, 65000: 1, 65030: 2, 70000:3, 86000:4, 88000:5, 
                90000:1, 93000:1, 99000:1, 70032:3, 74000:3, 80000:3}
    print(f"====== 岩性编码映射字典:{code2int}; 注意：若岩性不同，则需要更改映射字典")

    # 应用映射
    dfdata[lith_col] = dfdata[lith_col].map(code2int).astype(int)

    return dfdata

def lith_code_remap_int_yuanba_5class(dfdata, lith_col, lith_code_map_name):
    """
    将岩性编码重新映射到整数范围：
      0 -> 0
      1 -> 1 
      2 -> 2
      3 -> 3 
      4 -> 4
      5,6,7 -> 0
    """

    print("====== 将岩性编码重新映射到 [0, 4] 范围(五分类)!!!")
    
    # 构建初始映射字典
    code2int = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 0, 7: 0}
    print(f"====== 岩性编码映射字典:{code2int}; 注意：若岩性不同，则需要更改映射字典")

    # 应用映射
    dfdata[lith_col] = dfdata[lith_col].map(code2int).astype(int)

    return dfdata



def well_location_mask(dfwells, sam_len, args):
    """
    构建井位置mask (在两井连接处设置mask，以避免跨井制作样本)

    param:
    dfwells:所有井的df数据
    sam_len:构建的样本的长度

    return: 
    well_maskc: 在井接口处为0值的掩码
    """
    wells = dfwells[args.wl[0]].copy()  #注意只取一列则现在well是Series而不是DataFrame
    well_maskc = np.ones((dfwells.shape[0],1))   #maskc定义为mask_cat_loc

    wells_name = dfwells[args.wl[0]].unique()
    wells_num = len(wells_name)
    loc_li = [-1]  #loc_li存储完井位置,加上-1为表示起始处，0不可，因为起始点可能选择到0
    for i in range(wells_num):
        loc = dfwells[dfwells[args.wl[0]]==wells_name[i]].index[-1].tolist()  #index取[-1]即为该井完井位置
        #print('loc',loc)
        loc_li.append(loc)
        well_maskc[loc-sam_len+1 : loc+1]=0

    return well_maskc, loc_li


        # if self.mode != 'test':  ##训练和验证时，输入需要随机掩码
        #     # BERT-style 随机掩码（连续时间步）
        #     B, L, D = 1, fea_ori.shape[0], fea_ori.shape[1]  # 单样本
        #     valid_time_step = (masko.sum(dim=-1) == D)  # [L]  判断整行（时间步）是否在所有通道上都“完整”。
        #     mask_len = max(int(valid_time_step.sum().item() * self.mask_ratio), 1)  #添加1最少也会掩码1个
        #     mask = torch.zeros(L, dtype=torch.bool)

        #     # block mask
        #     max_start = max(valid_time_step.sum().item() - self.block_size + 1, 1)  #计算“可供 block 起始的数量”，用于产生起始位置。
        #     n_block = (mask_len + self.block_size - 1) // self.block_size  #用来确定要随机选多少个连续块（向上取整）。
        #     valid_idx = torch.where(valid_time_step)[0]  #得到那些可用作掩码的时间步索引

        #     if len(valid_idx) > 0:
        #         start_idx = torch.randperm(max_start)[:n_block]
        #         for s in start_idx:
        #             blk = valid_idx[s : min(s + self.block_size, len(valid_idx))]
        #             mask[blk] = True
        #         # 截断到mask_len
        #         mask_idx = torch.where(mask)[0]
        #         if len(mask_idx) > mask_len:
        #             mask[mask_idx[mask_len:]] = False

        #     # 扩展到通道维度
        #     mask_expand = mask.unsqueeze(-1).expand(-1, D)
        #     # masked输入
        #     fea_log = fea_log.clone()
        #     fea_log[mask_expand] = 0

        #     maskd = mask_expand  

class BertDataset_by_sam(Dataset):
    def __init__(self, data, spec, sam_len, sam_num,  
                maskc, loc_li, args,
                ):
        self.signal = data  #这里data和self.signal是dataframe
        #self.well_names = data[args.wl[0]].unique()
        #self.spec = spec
        self.sam_len = sam_len
        self.sam_num = sam_num
        self.maskc = maskc
        self.loc_li = loc_li
        self.args  =args
        self.fea = args.fea
        self.lith = args.lith
        self.md = args.md
        self.overlap_len = args.overlap_len
        self.min_freq = args.log_freq_range[0]
        self.max_freq = args.log_freq_range[-1]

        self.mask_ratio = 0.2  #已经试过了0.15概率  写0.2是为了和之前方案掩码整条曲线的比例相同
        self.block_size = 20
        self.min_gap = 20


        self.mode = args.load_mode if args.load_mode else "train"  #判断是在制作训练/验证集， 还是测试集
        print("====== data_loader中训练、验证样本的制作采取沿着井曲线采样的形式，非随机采样")
        print(f"====== 样本进行采样时，训练集和验证集设定了重叠长度{args.overlap_len}，以增加样本数量及曲线多样性,但测试集不重叠")

    def __len__(self):
        return self.sam_num
    
    def __getitem__(self, index):

        if self.mode != 'test':
            idx = index*self.overlap_len
            fea_ori, fea_np, lith_ori, masko, sam_stap, sam_endp, well_order, well_stap, well_endp, sam_md = self.sam_for_test(self.signal, idx)
        else:
            idx = index*self.sam_len
            fea_ori, fea_np, lith_ori, masko, sam_stap, sam_endp, well_order, well_stap, well_endp, sam_md = self.sam_for_test(self.signal, idx)


        # 转成 torch.Tensor
        fea_ori = torch.tensor(fea_ori, dtype=torch.float32)
        fea_log = torch.tensor(fea_np, dtype=torch.float32)
        lith_ori = torch.tensor(lith_ori, dtype=torch.long)   # label 用 long
        masko = torch.tensor(masko, dtype=torch.long)

        return {
                'fea_ori':fea_ori, 'fea_log':fea_log, 
                'lith': lith_ori,
		   		'masko': masko, "sam_md": sam_md,
				} 
         

    def make_random_mask(self, fea, masko, mask_ratio=0.15, block_size=20, min_gap=20):
        """
        生成 BERT 式 block 掩码
        fea: Tensor [seq_len, num_feat]
        masko: 初始有效掩码 [seq_len, num_feat]，0表示缺失，1表示有效
        mask_ratio: 总体掩码比例
        block_size: 每个连续掩码块的长度
        min_gap: block之间至少间隔多少个token
        """
        seq_len = fea.shape[0]
        total_mask_tokens = int(seq_len * mask_ratio)
        num_blocks = max(1, total_mask_tokens // block_size)

        maskd = np.ones_like(masko) 
        used = np.zeros(seq_len, dtype=bool)  # 标记哪些位置已被使用

        # 随机选择起始位置，但保证不重叠且有间隔
        starts = []
        attempts = 0
        while len(starts) < num_blocks and attempts < seq_len * 3:
            st = np.random.randint(0, seq_len - block_size + 1)
            ed = st + block_size
            # 检查是否与已有 block 重叠或间距不足
            if not used[max(0, st - min_gap):min(seq_len, ed + min_gap)].any():
                used[st:ed] = True
                starts.append(st)
            attempts += 1

        # 应用掩码
        for st in starts:
            maskd[st:st + block_size, :] = 0 

        return maskd




    def make_bert_mask_fast2(self, masko, mask_ratio=0.15, block_len=4, device=None):
        """
        向量化、优先掩码全通道有效位置的实现。
        masko: [seq_len, num_features] 初始掩码，1表示有效，0表示缺失
        mask_ratio: 总掩码比例（按 token 数计）
        block_len: 连续掩码长度（按 token）
        返回: mask_rand: [seq_len, num_features] 1表示有效，0表示掩码
        """
        if device is None:
            device = masko.device
        seq_len, num_features = masko.shape
        mask_rand = torch.ones_like(masko, dtype=torch.long, device=device)

        # 需要掩码的 token 数量（严格用 int）
        num_mask = int(seq_len * mask_ratio)
        if num_mask <= 0:
            return mask_rand

        # 全通道有效 / 含缺失 的索引
        valid_idx = torch.nonzero(masko.sum(dim=1) == num_features, as_tuple=True)[0]
        invalid_idx = torch.nonzero(masko.sum(dim=1) < num_features, as_tuple=True)[0]

        # 在各自集合内打乱，然后拼接（保证先用 valid，再用 invalid）
        if len(valid_idx) > 0:
            valid_idx = valid_idx[torch.randperm(len(valid_idx), device=device)]
        if len(invalid_idx) > 0:
            invalid_idx = invalid_idx[torch.randperm(len(invalid_idx), device=device)]
        candidate_idx = torch.cat([valid_idx, invalid_idx]) if (len(valid_idx) + len(invalid_idx)) > 0 else torch.tensor([], dtype=torch.long, device=device)
        if candidate_idx.numel() == 0:
            return mask_rand

        # 需要多少 block 起点：向上取整，保证覆盖足够
        num_blocks = max(1, math.ceil(num_mask / block_len))
        # 取前 num_blocks 个起点（若候选不够则取全部）
        start_points = candidate_idx[:num_blocks]

        # 生成 block 索引（向量化）
        offsets = torch.arange(block_len, device=device)  # [0,1,...,block_len-1]
        block_indices = (start_points[:, None] + offsets[None, :]).flatten()
        # 裁剪越界并去重
        block_indices = block_indices[block_indices < seq_len]
        if block_indices.numel() == 0:
            return mask_rand
        block_indices = torch.unique(block_indices)

        # 如果 block 产生的索引 > num_mask：随机删掉多余的（保持总数严格为 num_mask）
        if block_indices.numel() > num_mask:
            perm = torch.randperm(block_indices.numel(), device=device)
            chosen = block_indices[perm[:num_mask]]
        else:
            chosen = block_indices

        # 如果还不够 num_mask，用 candidate 中剩余位置（单点）补充
        if chosen.numel() < num_mask:
            remain = num_mask - chosen.numel()
            # 剩余候选来自 candidate_idx[num_blocks:]
            extra_pool = candidate_idx[num_blocks:]
            if extra_pool.numel() > 0:
                take = extra_pool[torch.randperm(extra_pool.numel(), device=device)[:remain]]
                chosen = torch.unique(torch.cat([chosen, take]))
            # 若仍然不足（极端情形：candidate 不够），则从所有位置随机补充（包含原来有缺失的位置）
            if chosen.numel() < num_mask:
                all_idx = torch.arange(seq_len, device=device)
                cand = all_idx[~torch.isin(all_idx, chosen)]
                if cand.numel() > 0:
                    extra_need = num_mask - chosen.numel()
                    take2 = cand[torch.randperm(cand.numel(), device=device)[:extra_need]]
                    chosen = torch.cat([chosen, take2])

        # 最终再次确保数量不超、不少
        if chosen.numel() > num_mask:
            chosen = chosen[torch.randperm(chosen.numel(), device=device)[:num_mask]]
        elif chosen.numel() < num_mask:
            # 兜底（极少发生）
            all_idx = torch.arange(seq_len, device=device)
            cand = all_idx[~torch.isin(all_idx, chosen)]
            need = num_mask - chosen.numel()
            chosen = torch.cat([chosen, cand[:need]])

        # 把这些 token 的所有通道置 0（掩码）
        mask_rand[chosen.long(), :] = 0

        return mask_rand



    def fea_lith_mask0(self,df):
        '''
        '''
        awmd = df[self.md].copy()   #awmd is all_wells_md
        awfea = df[self.fea].copy()   #awfea is all_wells_fea
        awlith = df[self.lith].copy()
        
        #选取随机起始点
        stap=np.random.randint(0,len(awfea)-self.sam_len)  #stap表示start_point
        #不选拼接处的数据
        while(self.maskc[stap]==0):
            stap=np.random.randint(0,len(awfea)-self.sam_len)
        endp=stap+self.sam_len  #endp表示end_point

        i=0
        while(self.loc_li[i]<stap):   
            ##判断选取到的样本属于哪口井，
            if stap<self.loc_li[i+1]:
                ##计算样本起始点、终止点的深度
                sam_stap = awmd.iloc[stap]
                sam_endp = awmd.iloc[stap + self.sam_len]    # - self.loc_li[i]
                
                ##判断井的起始点、终止点
                well_order = i
                well_stap = awmd.iloc[self.loc_li[i]+1]
                well_endp = awmd.iloc[self.loc_li[i+1]]
            i+=1

        fea_ori=awfea.iloc[stap:endp,:].copy()
        lith_ori = awlith.iloc[stap:endp,:].copy()

        ##将对应位置的参考样本选取出来，进而用参考样本的相关数值作均方归一化
        #fea, np_mean, np_std = self.mean_var_norm(fea_ori)

        spec_ori = self.spec[stap:endp,:].copy()
        spec_sam = np.where(np.isnan(spec_ori), 0, spec_ori)
        #制作初始的mask0   根据频谱获得
        mask0=np.where(np.isnan(spec_ori), 0, 1)  

        fea_np=np.where(pd.isnull(fea_ori), 0, fea_ori)

        lith_ori = lith_ori.values.squeeze(-1)  ##转变为numpy数据
        fea_ori = fea_ori.values
        return fea_ori, fea_np, lith_ori, spec_sam, mask0, sam_stap, sam_endp, well_order, well_stap, well_endp, #np_mean, np_std



    def sam_for_test(self, df, idx):
        awmd = df[self.md].copy()   #awmd is all_wells_md
        awfea = df[self.fea].copy()   #awfea is all_wells_fea
        awlith = df[self.lith].copy()

        stap = idx
        endp=stap+self.sam_len  #endp表示end_point

        ##补位填充下，无实际意义
        sam_stap= sam_endp= well_order= well_stap= well_endp = 0

        fea_ori = awfea.iloc[stap:endp,:].copy()
        lith_ori = awlith.iloc[stap:endp,:].copy()
        sam_md = awmd.iloc[stap:endp].copy()
        ##频谱的处理
        #spec_ori = self.spec[stap:endp,:].copy()
        #spec_sam = np.where(np.isnan(spec_ori), 0, spec_ori)

        ##将对应位置的参考样本选取出来，进而用参考样本的相关数值作均方归一化
        ##fea, np_mean, np_std = self.mean_var_norm(fea_ori)

        #制作初始的mask0   根据曲线获得
        masko=np.where(pd.isnull(fea_ori), 0, 1)

        fea_np=np.where(pd.isnull(fea_ori), 0, fea_ori).astype(np.float32)
        lith_ori = lith_ori.values.squeeze(-1).astype(np.int64)  ##转变为numpy数据，注意lith_ori需要本身就是完整的
        fea_ori = fea_ori.values.astype(np.float32)
        sam_md = sam_md.values.squeeze(-1).astype(np.float32)

        return fea_ori, fea_np, lith_ori, masko, sam_stap, sam_endp, well_order, well_stap, well_endp, sam_md


    def mean_var_norm(self, label):
        df1_mean=label.mean()
        df1_mean=np.where(pd.isnull(df1_mean), 0, df1_mean)
        #print('df1_mean',df1_mean.shape,df1_mean)
        df1_std=label.std(ddof=0)  #避免std使用index=0的影响
        df1_std=np.where(pd.isnull(df1_std), 0, df1_std)
        #print('df1_std.shape',df1_std.shape, df1_std)
        label_mean_var_norm=(label-df1_mean)/df1_std
        
        #由于进行gardner比较时，需要用到有实际物理意义的数据，所以这里返回numpy格式的均值和方差
        #直接np.array(df1_mean)时，其shape=(4,) 是一维数组
        np_mean = np.array(df1_mean).reshape((1,len(self.fea)))   #np_mean.shape=(1,4)
        np_std = np.array(df1_std).reshape((1,len(self.fea)))
        return label_mean_var_norm, np_mean, np_std
    





class Dataset_regression_v1(Dataset):
    """
    每个样本单独计算频谱，比较耗时
    """
    def __init__(self, data, sam_len, sam_num,  
                maskc, loc_li, args,
                ):
        self.signal = data  #这里data和self.signal是dataframe
        self.sam_len = sam_len
        self.sam_num = sam_num
        self.maskc = maskc
        self.loc_li = loc_li
        self.args  =args
        self.fea = args.fea
        self.lith = args.lith
        self.md = args.md
        self.min_freq = args.log_freq_range[0]
        self.max_freq = args.log_freq_range[-1]
        self.mode = args.load_mode if args.load_mode else "train"  #判断是在制作训练/验证集， 还是测试集
        
    def __len__(self):
        return self.sam_num

    
    def __getitem__(self, index):

        if self.mode != 'test':
            fea_ori, lith_ori, fea, mask0, sam_stap, sam_endp, well_order, well_stap, well_endp,np_mean, np_std=self.fea_lith_mask0(self.signal)
        else:
            idx = index*self.sam_len
            fea_ori, lith_ori, fea, mask0, sam_stap, sam_endp, well_order, well_stap, well_endp,np_mean, np_std = self.sam_for_test(self.signal, idx)
            
        #print('===== fea.shape: 应是(128,len(fea))', fea.shape)

        ## 提取每类曲线的频谱信息
        feas_spec = self.logs_calculate_spec(fea, mask0)  # # (128, 7*num_scales)

        ## 转置为：行表示通道数，列表示深度采样点，符合transformer的输入/token逻辑
        ## 注意，不要转置，输入应是 bsize * seqlen(token数目)(128) * feature(embedding)
        # feas_spec = feas_spec.T
        # mask0 = mask0.T
        # lith_ori = lith_ori.T

        # 转成 torch.Tensor
        feas_spec = torch.tensor(feas_spec, dtype=torch.float32)
        lith_ori = torch.tensor(lith_ori, dtype=torch.long)   # label 用 long
        mask0 = torch.tensor(mask0, dtype=torch.long)

        return {'fea':feas_spec, 'fea_ori':fea_ori,
                'lith': lith_ori,
		   		'mask0': mask0,
				} 
         

    def fea_lith_mask0(self,df):
        '''
        '''
        awmd = df[self.md].copy()   #awmd is all_wells_md
        awfea = df[self.fea].copy()   #awfea is all_wells_fea
        awlith = df[self.lith].copy()
        
        #选取随机起始点
        stap=np.random.randint(0,len(awfea)-self.sam_len)  #stap表示start_point
        #不选拼接处的数据
        while(self.maskc[stap]==0):
            stap=np.random.randint(0,len(awfea)-self.sam_len)
        endp=stap+self.sam_len  #endp表示end_point

        i=0
        while(self.loc_li[i]<stap):   
            ##判断选取到的样本属于哪口井，
            if stap<self.loc_li[i+1]:
                ##计算样本起始点、终止点的深度
                sam_stap = awmd.iloc[stap]
                sam_endp = awmd.iloc[stap + self.sam_len]    # - self.loc_li[i]
                
                ##判断井的起始点、终止点
                well_order = i
                well_stap = awmd.iloc[self.loc_li[i]+1]
                well_endp = awmd.iloc[self.loc_li[i+1]]
            i+=1

        fea_ori=awfea.iloc[stap:endp,:].copy()
        lith_ori = awlith.iloc[stap:endp,:].copy()

        ##将对应位置的参考样本选取出来，进而用参考样本的相关数值作均方归一化
        fea, np_mean, np_std = self.mean_var_norm(fea_ori)

        #制作初始的mask0
        mask0=np.where(pd.isnull(fea), 0, 1)  
        fea_np=np.where(pd.isnull(fea), 0, fea)

        lith_ori = lith_ori.values.squeeze(-1)  ##转变为numpy数据
        fea_ori = fea_ori.values
        return fea_ori, lith_ori, fea_np, mask0, sam_stap, sam_endp, well_order, well_stap, well_endp, np_mean, np_std



    def sam_for_test(self, df, idx):
        '''
        '''
        awmd = df[self.md].copy()   #awmd is all_wells_md
        awfea = df[self.fea].copy()   #awfea is all_wells_fea
        awlith = df[self.lith].copy()
        
        stap = idx
        endp=stap+self.sam_len  #endp表示end_point

        # i=0
        # while(self.loc_li[i]<stap):   
        #     ##判断选取到的样本属于哪口井，
        #     if stap<self.loc_li[i+1]:
        #         ##计算样本起始点、终止点的深度
        #         sam_stap = awmd.iloc[stap]
        #         sam_endp = awmd.iloc[stap + self.sam_len]    # - self.loc_li[i]
                
        #         ##判断井的起始点、终止点
        #         well_order = i
        #         well_stap = awmd.iloc[self.loc_li[i]+1]
        #         well_endp = awmd.iloc[self.loc_li[i+1]]
        #     i+=1
        ##补位填充下，无实际意义
        sam_stap= sam_endp= well_order= well_stap= well_endp = 0

        fea_ori=awfea.iloc[stap:endp,:].copy()
        lith_ori = awlith.iloc[stap:endp,:].copy()

        ##将对应位置的参考样本选取出来，进而用参考样本的相关数值作均方归一化
        fea, np_mean, np_std = self.mean_var_norm(fea_ori)

        #制作初始的mask0
        mask0=np.where(pd.isnull(fea), 0, 1)  
        fea_np=np.where(pd.isnull(fea), 0, fea)

        lith_ori = lith_ori.values.squeeze(-1)  ##转变为numpy数据
        fea_ori = fea_ori.values
        return fea_ori, lith_ori, fea_np, mask0, sam_stap, sam_endp, well_order, well_stap, well_endp, np_mean, np_std

    def mean_var_norm(self, label):
        df1_mean=label.mean()
        df1_mean=np.where(pd.isnull(df1_mean), 0, df1_mean)
        #print('df1_mean',df1_mean.shape,df1_mean)
        df1_std=label.std(ddof=0)  #避免std使用index=0的影响
        df1_std=np.where(pd.isnull(df1_std), 0, df1_std)
        #print('df1_std.shape',df1_std.shape, df1_std)
        label_mean_var_norm=(label-df1_mean)/df1_std
        
        #由于进行gardner比较时，需要用到有实际物理意义的数据，所以这里返回numpy格式的均值和方差
        #直接np.array(df1_mean)时，其shape=(4,) 是一维数组
        np_mean = np.array(df1_mean).reshape((1,len(self.fea)))   #np_mean.shape=(1,4)
        np_std = np.array(df1_std).reshape((1,len(self.fea)))
        return label_mean_var_norm, np_mean, np_std
    

    def time_freq_with_wavelet_transform(self, log):
        # 对每个样本进行小波变换
        ##scales：小波变换的尺度序列（你这里是 np.arange(20, 84)，即 [20, 21, ..., 83] 共 64 个尺度）
        scales = np.arange(self.min_freq, self.max_freq)  # 小波变换的尺度
        
        # 小波函数，避免 FutureWarning，使用 cmorB-C 形式
        wavelet_name = 'cmor1.5-1.0'  # B=1.5带宽，C=1.0中心频率
        ##wavelet：小波基，这里用 'cmor'（复 Morlet 小波，常用于时频分析）
        #freqs:每个尺度对应的小波中心频率（Hz）。维度：(len(scales),)=84-20
        coefs, freqs = pywt.cwt(log, scales, wavelet_name)
        coefs = np.abs(coefs).T
        #print("===== coefs.shape", coefs.shape)  #(sam_len, max_freq-min_freq)
        
        return coefs

    def logs_calculate_spec(self, logs, mask):
        """不同种类的测井曲线以此进行时频分析，且mask也同步相应扩充"""
        coefs_list = []

        for j in range(logs.shape[1]):
            l = logs[:, j]
            ## 取对应mask列
            mask_col = mask[:, j]  # (128,)
            l_coefs = self.time_freq_with_wavelet_transform(l) ## (128, num_scales)
            # 检查 NaN
            nan_l_coefs = np.isnan(l_coefs).sum()
            if nan_l_coefs > 0:
                #print("警告：l_coefs 中含有 NaN，已替换为 0")
                l_coefs = np.nan_to_num(l_coefs, nan=0.0)
            
            mask_bool = (mask_col != 0)  ## True 表示有效，False 表示要 mask 掉
            l_coefs[~mask_bool, :] = 0.0

            # mask_2d = mask_col[:, None]  ## (128, 1)，方便广播
            # l_coefs = l_coefs * mask_2d  # (128, num_scales)，掩码同步扩充

            coefs_list.append(l_coefs)

        # 按列拼接，每个 l_coefs 的 shape = (128, num_scales)
        logs_coefs = np.concatenate(coefs_list, axis=1)  # (128, 7*num_scales)

        return logs_coefs
    

    # def depth_loc(self,a):
    #     b = int((a-136.086)/0.152)  #元坝起始点深度3618.0，0.125；原norway136.086，0.152
    #     return b

    



