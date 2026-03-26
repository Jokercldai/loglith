"""resample_lith_by_seis_grid_time将岩性top、bot形式的数据转化成采样点形式
resample_lith_dense_to_sparse(),该函数将inline、xline、twt、lith四列密集数据抽稀，
使其稀疏符合地震/想要的分辨率"""




import numpy as np
import pandas as pd 
import os
from plot_fig import * 



def resample_lith_by_seis_grid_time(depth_data, lith_layer_with_depth_ranges
                                    ):
    """
    给定的岩性井数据是层段数据，如下
    twt_top	    twt_bot	    lith
    2174.2815	2179.4534	1
    2179.4534	2181.0986	0
    2181.0986	2184.7268	1
    2184.7268	2187.4424	0
    2187.4424	2194.4612	1
    将岩性段数据按照地震数据的网格采样点时间/深度重采样
    depth_data:深度采样点数据
    lith_layer_with_depth_ranges:层段的深度范围数据及岩性
    return C的格式为[深度采样点，岩性采样点]
    """

    # 初始化数组C
    C = []

    # 遍历数组B中的每一行
    for row in lith_layer_with_depth_ranges:
        start_depth, end_depth, lith = row
        print('start_depth, end_depth, lith', start_depth, end_depth, lith)

        # 找到在深度范围内的数据点
        indices = np.where((depth_data >= start_depth) & 
                           (depth_data <= end_depth))[0]
        print('indices', indices)
        
        # 将找到的深度数据和岩性数据添加到数组C中
        for index in indices:
            depth = depth_data[index]
            C.append([depth, lith])

    np_c = np.array(C)

    return np_c



def resample_lith_dense_to_sparse(file_path, modify_path, sam_rate, 
                                  plot_contrast_well=False):
    filename = os.path.basename(file_path).split('.xlsx')[0]
    #print('filename', filename)
    # 读取数据
    df = pd.read_excel(file_path, header=None, names=["inline", "xline", "twt", "lith"])
    #print('原始岩性数据', df)
    plot_ori_lith = df.iloc[:, 2:]
    plot_ori_lith = np.array(plot_ori_lith)

    # 计算新的索引
    min_twt = df['twt'].min()
    max_twt = df['twt'].max()
    new_index =pd.Series(np.arange(np.ceil(min_twt * (1/sam_rate)) / (1/sam_rate), 
                                   np.floor(max_twt * (1/sam_rate)) / (1/sam_rate) + sam_rate, 
                                 sam_rate))
    # 创建新的 DataFrame 包含新的采样时间点
    df_new_twt = pd.DataFrame({'twt': new_index})
    #print('df_new_twt', df_new_twt)

    # 使用 merge_asof() 函数将原始 DataFrame 和新的时间点 DataFrame 按照最接近(nearest)的时间点进行合并
    merged_df = pd.merge_asof(df_new_twt, df.sort_values('twt'), on='twt', direction='nearest')
    # 填充 inline、xline 和 lith 列的数据
    # merged_df['inline'] = merged_df['inline'].ffill()
    # merged_df['xline'] = merged_df['xline'].ffill()
    # merged_df['lith'] = merged_df['lith'].ffill()
    #print('merged_df', merged_df)

    df_resampled = merged_df.reindex(columns=['inline', 'xline', 'twt', 'lith'])
    #print('df_resampled', df_resampled)

    plot_resample_lith = df_resampled.iloc[:, 2:]
    plot_resample_lith = np.array(plot_resample_lith)

    if plot_contrast_well:
        plot_oriwell_and_resamplewell(plot_ori_lith, plot_resample_lith, 
                                    well_name=filename)

    well = np.array(df_resampled)
    # 将数据保存到新的文件中
    np.savetxt(modify_path+filename+'.txt', well, fmt=('%d', '%d', '%.2f', '%d'))
    


