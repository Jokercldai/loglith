"""
plot_filter_and_ori_lith  在同一图中分别绘制平滑前后的两个一维岩性
plot_feature_map  绘制线状特征图
plot_orilith_reslith 绘制原始的岩性和重采样之后的岩性 (整数)

visplot_multiple_3d():同时自适应绘制任意个3d图像

visplot_multiple_3d_V2:visplot_multiple_3d进阶版,设定cbar,最大最小值,cmap等 
注意：由于viserplot等库的不完善，暂时visplot_multiple_3d_V2还不能run,
visplot_multiple_3d_V2函数我觉着写的很正确，符合我的要求，饭还没有测试，后续考虑使用cigvis库来测试

visplot_3d_V2: 在jupyter中利用viserplot可视化三维体

visualize_logs_preds_targets: 可视化缺失曲线补全的代码

plot_bar:绘制每口井的预测指标的条形图
save_confusion_matrix_heatmap:绘制岩性预测和标签之间的混淆矩阵热力图
plot_heatmap:绘制每口井不同岩性结果的预测的热力图

"""


import numpy as np  #看看像numpy，pandas这种官方的缩写都是歌两个字母来缩写，可借鉴其写法（  它是按发音缩写的）
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
import scipy.stats as stats
import cigvis
from cigvis import viserplot # 导入package
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
import torch
import os
# sys.path.append('/home/cldai/sciresearch/resevior_pred/invdata4DCL/data/invCode/syn_pretrain/toolkit')
from .preprocess_data_1 import *


def plot_crossfig(df):
    # 假设你已经有一个 DataFrame，列包含：'Impedance', 'RMS', 'Sweetness', 'Gamma', 'Envelope', 'Phase', 'Frequency', 'Lithology'
    # Lithology = 0（泥岩）, 1（砂岩）

    # 例：波阻抗 vs RMS + 岩性标注
    plt.figure(figsize=(10,9), dpi=300)
    #sns.scatterplot(data=df, x='Impedance', y='RMS', hue='Lithology', palette='Set1')
    g = sns.pairplot(df[['impedance', 'rms', 'freqency', 'sweet', 'lith']], hue='lith')

    # 设置 xlabel 和 ylabel 的字体大小
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontsize=15)
            ax.set_ylabel(ax.get_ylabel(), fontsize=15)
            #ax.tick_params(labelsize=10)  # 坐标刻度字体大小

    # 设置图例字体
    if g._legend:
        g._legend.set_title("Lith")
        g._legend.get_title().set_fontsize(20) 
        for text in g._legend.texts:
            text.set_fontsize(15)
    plt.show()



def visplot_multiple_3d(data_list, cmap_list, share_camera=True):
    """同时绘制多个3d图像    不可以，viser暂时没有开放这个，cigvis可以同时绘制多个函数"""
    # 创建节点列表
    nodes_list = [viserplot.create_slices(data, cmap=cmap_list[i]) 
                  for data, i in enumerate(data_list)]

    # 计算网格大小
    n_plots = len(data_list)
    rows = int(np.ceil(n_plots**0.5))  #np.ceil向上取整
    cols = int(np.ceil(n_plots / rows)) 

    # 绘图
    cigvis.plot3D(
        nodes_list,
        grid=(rows, cols),  # 动态定义网格
        share=share_camera,  # 链接所有相机
        size=(1000, 800),
        #savename='example.png'
    )


def visplot_multiple_3d_V2(data_list, cmap_list=['gray'], 
                           cmin_list=None, cmax_list=None, 
                           return_cbar_list=None, 
                           share_camera=True):
    """
    同时绘制多个3D图像，并可为每个图像添加cbar和最大最小值。
    不可以，viser暂时没有开放这个，cigvis可以同时绘制多个函数
    
    参数：
    - data_list: 数据列表
    - cmap_list: 每个数据的cmap列表,默认均用gray
    - cmin_list: 每个数据的cmin列表（可以为None）
    - cmax_list: 每个数据的cmax列表（可以为None）
    - return_cbar_list: 是否为每个数据添加cbar的布尔列表
    - share_camera: 是否共享相机视角
    """
    # 如果cmin_list或cmax_list未定义，则设置为None
    if cmin_list is None:
        cmin_list = [None] * len(data_list)
    if cmax_list is None:
        cmax_list = [None] * len(data_list)
    if return_cbar_list is None:
        return_cbar_list = [False] * len(data_list)

    if len(cmap_list) == 1:
        cmap_list = cmap_list * len(data_list)


    nodes_list = []
    for i, data in enumerate(data_list):
        cmap = cmap_list[i]
        cmin = cmin_list[i] if cmin_list[i] is not None else np.nanmin(data)
        cmax = cmax_list[i] if cmax_list[i] is not None else np.nanmax(data)
        return_cbar = return_cbar_list[i]

        if return_cbar:
            nodes, cbar = viserplot.create_slices(data, cmap=cmap, clim=[cmin, cmax], #cmin=cmin, cmax=cmax, 
                                               return_cbar=True)
            nodes_list.append(nodes)
            nodes_list.append([cbar])
        else:
            nodes = viserplot.create_slices(data, cmap=cmap, clim=[cmin, cmax])  #,cmin=cmin, cmax=cmax
            nodes_list.append(nodes)

    # 展平节点列表
    flat_nodes_list = [node for nodes in nodes_list for node in nodes]

    # 计算网格大小
    n_plots = len(flat_nodes_list)
    rows = int(np.ceil(n_plots ** 0.5))
    cols = int(np.ceil(n_plots / rows))

    # 绘图
    viserplot.plot3D(
        flat_nodes_list,
        grid=(rows, cols),  # 动态定义网格
        share=share_camera,  # 链接所有相机
        size=(1000, 800),
        # savename='example.png'
    )



def plot_sample_distribution_heatmap(data, save_fig=None):
    # 将 NumPy 数组转换为 Pandas DataFrame
    df = pd.DataFrame(data, columns=['inline', 'xline', 'twt', 'count'])

    # 为每口井生成唯一的井 ID（根据 inline 和 xline）
    df['well_id'] = df.groupby(['inline', 'xline'], sort=False).ngroup()

    # 创建一个透视表，以便在热力图中使用
    heatmap_data = df.pivot_table(index='twt', columns='well_id', values='count', fill_value=0)

    # 绘制热力图
    plt.figure(figsize=(12, 8), dpi=300)
    sns.heatmap(heatmap_data, cmap='BuPu', cbar_kws={'label': 'Count'})  #YlGnBu
    plt.xlabel('Well ID')
    plt.ylabel('TWT/Depth')
    plt.title('Heatmap of sample distribution')
    if save_fig is not None:
        plt.savefig(save_fig
	      + 'train_sam_distribution'+'.jpg', dpi=300, bbox_inches='tight')
    plt.show()



def plot_orilith_reslith(ori_lith, res_lith, wname=None):
    fig,axs = plt.subplots(1, 2, figsize=(7, 10), sharey=True)  #dpi=300, 

    axs[0].plot(ori_lith[:,1], ori_lith[:,0], color='black')
    axs[1].plot(res_lith[:,1], res_lith[:,0], color='black')
    axs[0].set_xlabel('Original Lith', fontsize=15)
    axs[1].set_xlabel('Resampled Lith', fontsize=15)
    axs[0].set_ylabel('Depth(m)', fontsize=15)

    axs[0].invert_yaxis()

    if wname is not None:
        fig.suptitle(wname, fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_orilith_reslith_grori_grnormed(ori_lith, res_lith, gr_ori, 
                                        gr_normed, wname=None):
    fig,axs = plt.subplots(1, 4, figsize=(7, 10), sharey=True)  #dpi=300, 

    axs[0].plot(ori_lith[:,1], ori_lith[:,0], color='black')
    axs[1].plot(res_lith[:,1], res_lith[:,0], color='black')
    axs[2].plot(gr_ori, res_lith[:,0], color='black')
    axs[3].plot(gr_normed, res_lith[:,0], color='black')

    axs[0].set_xlabel('Original Lith', fontsize=15)
    axs[1].set_xlabel('Resampled Lith', fontsize=15)
    axs[2].set_xlabel('gr_ori', fontsize=15)
    axs[3].set_xlabel('gr_normed', fontsize=15)
    axs[0].set_ylabel('Depth(m)', fontsize=15)

    axs[0].invert_yaxis()

    if wname is not None:
        fig.suptitle(wname, fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_filter_and_ori_lith(x1,x2,label1=None, label2=None):

    # 创建一个图形对象
    plt.figure(figsize=(4,15))
    y = np.arange(len(x1))
    # 绘制第一条曲线
    plt.plot(x1, y, label=label1, color='b', linestyle='-', linewidth=2)
    # 绘制第二条曲线
    plt.plot(x2, y, label=label2, color='r', linestyle='--', linewidth=2)
    # 添加图例
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(len(x1), 0)
    plt.tight_layout()
    plt.show()


def plot_feature_map(x1, x2, figname=None, figpath=None):
    nums = 2
    y = np.arange(0, x1.shape[1])
    print('x1.shape', x1.shape, 'y.shape', y.shape)
    fig, ax = plt.subplots(1, nums, figsize=(nums*5,10), dpi=300, sharey = True)
    for i in range(x1.shape[0]):
        ax[0].plot(x1[i,:], y)  
        ax[1].plot(x2[i,:], y)             
    ax[0].set_title('Fea_map1', fontsize=15)
    ax[1].set_title('Fea_map2', fontsize=15)
    ax[0].set_ylabel("Sample", fontsize=15)
    ax[0].set_xlabel("X1", fontsize=15)
    ax[1].set_xlabel("X2", fontsize=15)
    if figname is not None:
        fig.suptitle(str(figname), fontsize=20)
    plt.tight_layout()
    if (figpath and figname) is not None:
        plt.savefig(figpath+figname+'.png',
                dpi=300,bbox_inches='tight')
    plt.show()

def plot_TSNE_2components(tsne1, tsne2):
    nums = 2
    fig, ax = plt.subplots(1, nums, figsize=(nums*5,10), dpi=300)
    ax[0].scatter(tsne1[:, 0], tsne1[:, 1]) 
    ax[1].scatter(tsne2[:, 0], tsne2[:, 1])    

    ax[0].set_title('TSNE_map1', fontsize=15)
    ax[1].set_title('TSNE_map2', fontsize=15)

    ax[0].set_xlabel("Component 1", fontsize=15)
    ax[0].set_ylabel("Component 2", fontsize=15)
    
    ax[1].set_xlabel("Component 1", fontsize=15)
    ax[1].set_ylabel("Component 2", fontsize=15)

    plt.tight_layout()
    plt.show()


def plot_score_matrix(matrix,figname=None):
    fig = plt.figure(figsize = (7,7), dpi=300)
    sn.heatmap(matrix, annot=True, cmap="BuPu",fmt='.2f', annot_kws={"fontsize":10})
    plt.title(figname+"Attention coefficient",fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    #plt.savefig(args.fig_path+'my_qt_eva/'+figname+'.png',dpi=300,bbox_inches='tight')


def plot_oriwell_and_resamplewell(ori_well, resample_well, well_name=None):
    nums=2
    fig, ax = plt.subplots(1, nums, figsize=(nums*3,10), dpi=300, sharey = False)

    ax[0].plot(ori_well[:,1], ori_well[:,0], color='b')  
    ax[0].set_title('Original lith', fontsize=15)
    ax[1].plot(resample_well[:, 1], resample_well[:, 0], color='b')  
    ax[1].set_title('Resampled lith', fontsize=15)
    #print('11111', ori_well, resample_well)
    for i in range(nums):
        ax[i].set_xlabel("Lith", fontsize=15)
        ax[i].set_ylabel("TWT", fontsize=15)

    if well_name is not None:
        fig.suptitle(str(well_name), fontsize=20)
    plt.tight_layout()
    plt.show()

    

###下面code是不可以的，因为这个函数在jupyter中没法运行，此外，没法通过py程序保存图，因为没法保存
def jupyter_cigvis_mutiple_3d(data, far, mid, near, enve, freq, phase,imp,rgt, 
                              spec0,spec1,spec2,spec3,
                              grid=None,share=True, 
                              cmin=None, cmax=None, 
                              cmap='Petrel', cbar_label='rgt', 
              cbar=True, save_dir=None, save_name=None):
    if (cmin != None) & (cmax != None):
        cmin=cmin
        cmax=cmax
    else:
        cmin=np.nanmin(data)
        cmax=np.nanmax(data)
        print('自动除却nan后数据的cmin, cmax', cmin, cmax)

    nodes3 = cigvis.create_slices(data, cmap=cmap,)  #  cbar
    
    nodes4 = cigvis.create_slices(far, cmap=cmap, 
                    )  #  cbar
    nodes5 = cigvis.create_slices(mid, cmap=cmap, 
                    )  #  cbar
    nodes6 = cigvis.create_slices(near, cmap=cmap,
                    )  #  cbar
    nodes7 = cigvis.create_slices(enve, cmap=cmap, 
                    )  #  cbar
    nodes8 = cigvis.create_slices(freq, cmap=cmap, 
                    )  #  cbar
    nodes9 = cigvis.create_slices(phase, cmap=cmap, 
                    )  #  cbar
    nodes10 = cigvis.create_slices(imp, cmap=cmap,
                    )  #  cbar
    nodes11 = cigvis.create_slices(rgt, cmap=cmap, 
                    )  #  cbar
    nodes12 = cigvis.create_slices(spec0, cmap=cmap, 
                    )  #  cbar
    nodes13 = cigvis.create_slices(spec1, cmap=cmap, 
                    )  #  cbar
    nodes14 = cigvis.create_slices(spec2, cmap=cmap,
                    )  #  cbar
    nodes15 = cigvis.create_slices(spec3, cmap=cmap, 
                    )  #  cbar
    """"""

    cigvis.plot3D(nodes3,
                  size=(1000, 800),
                  grid=grid, #share=share,
                savedir= save_dir,
                savename=save_name)
    
    """[nodes3, nodes4,nodes5, nodes6, nodes7, nodes8, nodes9,
                  nodes10, nodes11, nodes12, nodes13, nodes14],"""
    








def plot_2Dprofile_and_well(seis, well, iline=0, xline=0, 
                            plot_section='inline', alpha=0.5, well_name=None):
    ##绘制三维数据体的二维剖面以及其上的对应的部分井
    ##绘制地震二维剖面，叠加测井二维自制剖面，检验井震是否匹配
    
    colors_bar = ['yellow', 'red', 'blue']
    label_bar = ['1', '2', '3']
    mycmap_bar = ListedColormap(colors_bar)
    
    
    d = seis.copy()
    d = d.transpose()
    print('转置后数据的维度', d.shape)
    print('数据最大值、最小值：', np.max(d), np.min(d))

    il = d[:, :, iline]
    xl = d[:, xline, :]
    
    if plot_section == 'inline':
        print('绘制 inline 剖面！')
        dat = il.copy()
        ##绘制inline剖面，则将well的Xline号投影到剖面上
        line = well[0,1]
    else:
        dat = xl.copy()
        print('绘制 xline 剖面！')
        line = well[0,0]

    well_profile = np.full(dat.shape, np.nan)
    lith = well[:,3]  #3列是岩性列

    sta = well[0,2]
    endp = well[-1,2]+1
    for i in range(-5, 5):
        well_profile[sta:endp, line+i] = pd.Series(lith).values

    ##查看井中有哪些岩性，以便选择出对应的标签及岩性
    a = np.unique(lith)
    print('井中有哪些岩性编号：', a)
    colors = []
    for i in range(len(a)):
        c = a[i]
        print('c', c)
        colors.append(colors_bar[c-1])
    print('井中岩性对应的颜色', colors)
    mycmap = ListedColormap(colors)

    # 绘制第一个剖面
    plt.imshow(dat, cmap='jet', #extent=(dat.min(), dat.max(), lith.min(), lith.max()), 
               origin='upper', alpha=1)
    # 绘制第二个剖面叠加在第一个剖面上
    print('查看井编辑的剖面是否全为nan', np.all(np.isnan(well_profile)))
    plt.imshow(well_profile, cmap=mycmap,  #'plasma', 
               #extent=(dat.min(), dat.max(), lith.min(), lith.max()), 
               origin='upper', alpha=alpha)

    # 创建一个新的轴用于colorbar
    cax = plt.axes([1, 0.1, 0.03, 0.8])  # 调整这些参数以适应你的布局
    # 创建colorbar，并在每个刻度中间添加对应的编码
    colorbar = ColorbarBase(cax, cmap=mycmap_bar, 
                            ticks=np.arange(0,1, 1/len(colors_bar))+0.5/len(colors_bar),  
                            orientation='vertical') 
    colorbar.set_ticklabels(label_bar)

    plt.savefig('/home/cldai/sciresearch/resevior_pred/figure/2Dprofile_and_well/'+
                well_name+'_'+plot_section+
                '_2Dprofile_and_well.jpg', dpi = 500, bbox_inches= 'tight')
    plt.show()



def plot_lith_pie(categolies, lith_cate_num, dataset_name, cate_names=None, colors=None):
    """
    categolies：数组形式， 如[0 1 2 5 6] 
    lith_cate_num：数组形式， 如[1464 1486   18   19    4]
    """
    ##绘制岩性种类的扇形图
    # 示例数据
    
    sizes = lith_cate_num
    #categolies = ['mudstone(1)', 'sandstone(2)', 'limestone(3)']
    if cate_names is not None:
        labels = [f"{cate_name}_{category}\n{size}" for category, size, cate_name in zip(
            categolies, sizes, cate_names)]
    else:
        labels = [f'{category}\n{size}' for category, size in zip(categolies, sizes)]
    
    #colors = ['lightcoral', 'lightskyblue', 'lightgreen']

    # 绘制扇形图
    fig = plt.figure(dpi = 300)  #facecolor='white',底图
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)  #autopct=  None

    # 设置图形标题
    plt.title(f'Lithology distribution of the {dataset_name}', fontsize=15, 
              pad=20)  # 使用 'pad' 参数来设置标题与图之间的距离
    # 显示图形
    plt.axis('equal')  # 保持纵横比一致
    plt.legend(labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_input_imshow_label(inputxlabel=None, seis=None, far=None, mid=None, near=None,
                            interp=None, rgt=None, phase=None, 
                         freq=None, ampli=None, imp=None, spec=None, log=None,
                         plot_label=False, label_over=False, label_type='imshow',
                         lithcode='-101',
                         well_name=None, codelabel=None):
    """
    绘制完整的抽取的trace,包括二维的频谱
    以及label

    num: 输入数据类型的数目
    """

    ##砂岩红色，泥岩绿色，灰岩黄色
    if lithcode=='012':
        colors=['green', 'red', 'yellow']
        codeticks = [0,1,2]
        bounds = [-0.5, 0.5, 1.5, 2.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    if lithcode=='123':
        colors=['green', 'red', 'yellow']
        codeticks = [1,2,3]
        bounds = [0.5, 1.5, 2.5, 3.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    if lithcode=='-101':
        colors=['yellow', 'green', 'red']
        codeticks = [-1,0,1]
        bounds = [-1.5, -0.5, 0.5, 1.5]
        codelabels = ['limestone', 'mudstone', 'sandstone']

    if codelabel is not None:
        codelabels= codelabel

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))

    log2d = log[:, np.newaxis]

    #inputxlabel = [seis, interp, rgt, phase, freq, ampli, imp, spec]
    ori_data = [seis, far, mid, near, interp, rgt, phase, freq, ampli, imp, spec]
    data = [x for x in ori_data if x is not None]

    ##判断是否需要绘制label
    if plot_label == True:
        nums = len(inputxlabel)+1
    else:
        nums = len(inputxlabel)

    fig, ax = plt.subplots(1, nums, figsize=(nums*3,10), dpi=300, sharey = True)
    if well_name is not None:
        fig.suptitle(str(well_name), fontsize=20)
    y = np.arange(len(seis))
    

    if plot_label == False:
        for i in range(nums-1):
        ##这里减一是为了能够留出最后一列画频谱图
            ax[i].plot(data[i], y, color='b')  
            ax[i].set_ylim(len(seis), 0)
            ax[i].set_xlabel(inputxlabel[i], fontsize=20)
            #ax[0].tick_params(axis='x', labelrotation=45)

        ax_1 = ax[-1].imshow(spec, cmap='jet', aspect='auto')
        ax[-1].set_xlabel(inputxlabel[-1], fontsize=20)
        fig.colorbar(ax_1, ax = ax[-1])    

    else:
       
        for i in range(nums-2):
        ##这里减一是为了能够留出最后一列画频谱图
            if label_over==True:
                ##当label_over==True时，每列子图label和输入曲线画在一起
                ax[i].imshow(log2d, cmap=cmap, norm=norm, 
                             interpolation='nearest', aspect='auto',
                             )  #extent=[data[i].min(), data[i].max(), y.min(), y.max()]
                #print('每列子图label和输入曲线画在一起')   
            ax[i].plot(data[i], y, color='b') 
            
            ax[i].set_ylim(len(seis), 0)
            ax[i].set_xlabel(inputxlabel[i], fontsize=20)
            #ax[0].tick_params(axis='x', labelrotation=45)

        ax_1 = ax[-2].imshow(spec, cmap='jet', aspect='auto')
        ax[-2].set_xlabel(inputxlabel[-1], fontsize=20)
        fig.colorbar(ax_1, ax = ax[-2])   
        
        if label_type=='imshow':
            im = ax[-1].imshow(log2d, cmap=cmap, norm=norm, 
                            interpolation='nearest', aspect='auto')
            ##使用 extend='both' 映射外溢的标签
            cbar = plt.colorbar(im, #extend='both', 
                                ticks=codeticks, #ticks指定了刻度
                                boundaries=bounds )  ## boundaries表示边界值   
            # 标注 colorbar 的标签
            #print('11111')
            cbar.set_ticklabels(codelabels, fontsize=15) 


        elif label_type == 'line':
            ax[-1].plot(log, y, color='r')  
            # ax[-1].set_ylim(len(log), 0)
            # ax[-1].set_xlabel('true lith', fontsize=20)
        ax[-1].set_ylim(len(log)-1, 0)
        ax[-1].set_xlabel('true lith', fontsize=20)

    plt.tight_layout() 
    plt.show()




    # #选择一种cmap
    # cmap = matplotlib.cm.get_cmap('rainbow')   #  viridis
    # #选取n种颜色
    # cmap = ListedColormap(cmap(np.linspace(0, 255, param['lith_class']).astype(np.uint8)))
    # # 创建边界规范。bounds 指定了颜色分界线在哪里
    # bounds = np.array(list(np.arange(param['lith_class']))+[param['lith_class']]) #需要+1，否则少一种颜色
    # norm = BoundaryNorm(bounds, cmap.N)
    # # 绘制颜色条
    # cb = plt.colorbar(
    # plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    # ticks=bounds+0.5, #ticks指定了刻度标在哪里，+0.5是为了使刻度标在颜色中间
    # boundaries=bounds,
    # orientation='vertical',
    # format='%d'
    # )




def plot_input_val_por(inputxlabel=None, seis=None, far=None, mid=None, near=None,
                            interp=None, rgt=None, phase=None, 
                         freq=None, ampli=None, imp=None, spec=None, 
                         val=None, por=None, seis_y=None, log_y=None, 
                         well_name=None):
    """
    绘制完整的抽取的trace,包括二维的频谱
    以及label

    num: 输入数据类型的数目
    """


    #inputxlabel = [seis, interp, rgt, phase, freq, ampli, imp, spec]
    ori_data = [seis, far, mid, near, interp, rgt, phase, freq, 
                ampli, imp, spec, val, por]
    data = [x for x in ori_data if x is not None]

    nums=len(inputxlabel)+2
    fig, ax = plt.subplots(1, nums, figsize=(nums*3,10), dpi=300, sharey = True)
    if well_name is not None:
        fig.suptitle(str(well_name), fontsize=20)
    

    for i in range(nums-3):
        #print('数据的维度', len(data[i]), len(seis_y))
        ax[i].plot(data[i], seis_y, color='b') 
        #ax[i].set_ylim(len(seis), 0)
        ax[i].set_xlabel(inputxlabel[i], fontsize=20)
        #ax[0].tick_params(axis='x', labelrotation=45)

    ax_1 = ax[-3].imshow(spec, cmap='jet', aspect='auto', 
                         extent=[0, spec.shape[1], seis_y[-1], seis_y[0]])
    ax[-3].set_xlabel(inputxlabel[-1], fontsize=20)
    fig.colorbar(ax_1, ax = ax[-3])   
    
    ax[-2].plot(val, log_y, color='b') 
    ax[-2].set_xlabel('Val', fontsize=20)
    ax[-1].plot(por, log_y, color='b') 
    ax[-1].set_xlabel('Por', fontsize=20)

    #ax[0].invert_yaxis()

    plt.tight_layout() 
    plt.show()





def plot_com_input_label(inputxlabel=None, seis=None, interp=None, rgt=None, phase=None, 
                         freq=None, ampli=None, imp=None, spec=None, log=None,
                         plot_label=False, label_over=False, label_type='line'):
    """
    绘制完整的抽取的trace,包括二维的频谱
    以及label

    num: 输入数据类型的数目s
    """

    #inputxlabel = [seis, interp, rgt, phase, freq, ampli, imp, spec]
    ori_data = [seis, interp, rgt, phase, freq, ampli, imp, spec]
    data = [x for x in ori_data if x is not None]

    ##判断是否需要绘制label
    if plot_label == True:
        nums = len(inputxlabel)+1
    else:
        nums = len(inputxlabel)

    fig, ax = plt.subplots(1, nums, figsize=(nums*3,10), dpi=300, sharey = True)
    y = np.arange(len(seis))
    

    if plot_label == False:
        for i in range(nums-1):
        ##这里减一是为了能够留出最后一列画频谱图
            ax[i].plot(data[i], y, color='b')  
            ax[i].set_ylim(len(seis), 0)
            ax[i].set_xlabel(inputxlabel[i], fontsize=20)
            #ax[0].tick_params(axis='x', labelrotation=45)

        ax_1 = ax[-1].imshow(spec, cmap='jet', aspect='auto')
        ax[-1].set_xlabel(inputxlabel[-1], fontsize=20)
        fig.colorbar(ax_1, ax = ax[-1])    

    else:
       
        for i in range(nums-2):
        ##这里减一是为了能够留出最后一列画频谱图
            ax[i].plot(data[i], y, color='b')  
            if label_over==True:
                ##当label_over==True时，每列子图label和输入曲线画在一起
                ax[i].plot(log, y, color='r') 
                #print('每列子图label和输入曲线画在一起')
            
            ax[i].set_ylim(len(seis), 0)
            ax[i].set_xlabel(inputxlabel[i], fontsize=20)
            #ax[0].tick_params(axis='x', labelrotation=45)

        ax_1 = ax[-2].imshow(spec, cmap='jet', aspect='auto')
        ax[-2].set_xlabel('freqency spectrum', fontsize=20)
        fig.colorbar(ax_1, ax = ax[-2])   
        if label_type == 'line':
            ax[-1].plot(log, y, color='r')  
        # elif  label_type == 'plane':
        #     ax[-1].imshow()
        ax[-1].set_ylim(len(log), 0)
        ax[-1].set_xlabel('true log', fontsize=20)
        #ax[0].tick_params(axis='x', labelrotation=45)

    plt.tight_layout() 
    



def plot1d_input_label(inputxlabel=None, seis=None, interp=None, rgt=None, phase=None, 
                         freq=None, ampli=None, imp=None, spec=None, log=None,
                         plot_label=False):
    """
    绘制完整的抽取的trace,包括二维的频谱
    以及label

    num: 输入数据类型的数目s
    """

    #inputxlabel = [seis, interp, rgt, phase, freq, ampli, imp, spec]
    ori_data = [seis, interp, rgt, phase, freq, ampli, imp, spec]
    data = [x for x in ori_data if x is not None]

    ##判断是否需要绘制label
    if plot_label == True:
        nums = len(inputxlabel)+1
    else:
        nums = len(inputxlabel)

    fig, ax = plt.subplots(1, nums, figsize=(nums*3,10), dpi=300, sharey = True)
    y = np.arange(len(seis))
    

    if plot_label == False:
        for i in range(nums):
        ##这里减一是为了能够留出最后一列画频谱图
            ax[i].plot(data[i], y, color='b')  
            ax[i].set_ylim(len(seis), 0)
            ax[i].set_xlabel(inputxlabel[i], fontsize=20)
            #ax[0].tick_params(axis='x', labelrotation=45)

        # ax_1 = ax[-1].imshow(spec, cmap='jet', aspect='auto')
        # ax[-1].set_xlabel(inputxlabel[-1], fontsize=20)
        # fig.colorbar(ax_1, ax = ax[-1])    

    else:
        #for i in range(nums-2):
        for i in range(nums-1):
        ##这里减一是为了能够留出最后一列画频谱图
            ax[i].plot(data[i], y, color='b')  
            ax[i].set_ylim(len(seis), 0)
            ax[i].set_xlabel(inputxlabel[i], fontsize=20)
            #ax[0].tick_params(axis='x', labelrotation=45)

        # ax_1 = ax[-2].imshow(spec, cmap='jet', aspect='auto')
        # ax[-2].set_xlabel('freqency spectrum', fontsize=20)
        # fig.colorbar(ax_1, ax = ax[-2])   

        ax[-1].plot(log, y, color='r')  
        ax[-1].set_ylim(len(log), 0)
        ax[-1].set_xlabel('true log', fontsize=20)
        #ax[0].tick_params(axis='x', labelrotation=45)

    plt.tight_layout() 
    plt.show()




def imshow_traces(int_pred, label, lithcode='-101',
                  savename=None, well_name=None,
                  codelable = None, twt=None):
    fig,axs = plt.subplots(1,2, sharey = True, figsize=(2.5*2,8), dpi=300)
    if well_name is not None:
        fig.suptitle(well_name, fontsize=20)
    print('注意岩性编码是', lithcode)

    ##砂岩红色，泥岩绿色，灰岩黄色
    if lithcode=='012':
        colors=['green', 'red', 'yellow']
        codeticks = [0,1,2]
        bounds = [-0.5, 0.5, 1.5, 2.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    if lithcode=='123':
        colors=['green', 'red', 'yellow']
        codeticks = [1,2,3]
        bounds = [0.5, 1.5, 2.5, 3.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    if lithcode=='-101':
        colors=['yellow', 'green', 'red']
        codeticks = [-1,0,1]
        bounds = [-1.5, -0.5, 0.5, 1.5]
        codelabels = ['limestone', 'mudstone', 'sandstone']

    if codelable is not None:
        codelabels = codelable

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))


    y1 = int_pred[:, np.newaxis]
    y2 = label[:, np.newaxis]
    im1 = axs[0].imshow(y1, cmap=cmap, norm=norm, 
                        interpolation='nearest', aspect='auto')
    #axs[-1].set_ylim(len(log)-1, 0)
    axs[0].set_xlabel('pred lith', fontsize=20)

    im2 = axs[1].imshow(y2, cmap=cmap, norm=norm, 
                        interpolation='nearest', aspect='auto')
    axs[1].set_xlabel('true lith', fontsize=20)

    # 设置纵坐标刻度和标签
    if twt is not None:
        yticks = np.arange(0, twt.shape[0], int(twt.shape[0]/10))  # 定义刻度位置
        yticklabels = [f'{i}' for i in twt[::int(twt.shape[0]/10)]]  # 定义刻度标签
        axs[0].set_yticks(yticks)
        axs[0].set_yticklabels(yticklabels)

    ##使用 extend='both' 映射外溢的标签
    cbar = plt.colorbar(im2, #extend='both', 
                        ticks=codeticks, #ticks指定了刻度
                        boundaries=bounds )  ## boundaries表示边界值   
    # 标注 colorbar 的标签
    cbar.set_ticklabels(codelabels, fontsize=15) 

    axs[0].set_ylabel('depth(m)')

    plt.tight_layout()
    if savename:
        plt.savefig(savename+'.jpg', dpi=300, bbox_inches='tight')
    plt.show()



def plot_2dprofile(int_pred, label, lithcode='-101',
                    inline=None, xline=None, twt=None,
                    savename='profile', section_name=None,
                    plot_section='inline'):

    fig,axs = plt.subplots(1,2, sharey = True, figsize=(7*2,8), dpi=300)
    if section_name is not None:
        fig.suptitle(section_name, fontsize=20)
    print('注意岩性编码是', lithcode)

    ##砂岩红色，泥岩绿色，灰岩黄色
    if lithcode=='012':
        colors=['green', 'red', 'yellow']
        codeticks = [0,1,2]
        bounds = [-0.5, 0.5, 1.5, 2.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    if lithcode=='123':
        colors=['green', 'red', 'yellow']
        codeticks = [1,2,3]
        bounds = [0.5, 1.5, 2.5, 3.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    if lithcode=='-101':
        colors=['yellow', 'green', 'red']
        codeticks = [-1,0,1]
        bounds = [-1.5, -0.5, 0.5, 1.5]
        codelabels = ['limestone', 'mudstone', 'sandstone']

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))

    pred_cube = int_pred.copy()
    pred_cube = pred_cube.transpose()
    label_cube = label.copy()
    label_cube = label_cube.transpose()

    if plot_section == 'inline':
        print('绘制 inline 剖面！')
        pred_sec = pred_cube[:, :, inline]
        true_sec = label_cube[:, :, inline]
    elif plot_section == 'xline':
        pred_sec = pred_cube[:, xline, :]
        true_sec = label_cube[:, xline, :]
        print('绘制 xline 剖面！')
    else:
        pred_sec = pred_cube[twt, :, :]
        true_sec = label_cube[twt, :, :]
        print('绘制 depth 剖面！')
    
    
    im1 = axs[0].imshow(pred_sec, cmap=cmap, norm=norm, 
                        interpolation='nearest', aspect='auto')
    #axs[-1].set_ylim(len(log)-1, 0)
    axs[0].set_xlabel('pred lith', fontsize=20)

    im2 = axs[1].imshow(true_sec, cmap=cmap, norm=norm, 
                        interpolation='nearest', aspect='auto')
    axs[1].set_xlabel('seismic', fontsize=20)

    ##使用 extend='both' 映射外溢的标签
    cbar = plt.colorbar(im2, #extend='both', 
                        ticks=codeticks, #ticks指定了刻度
                        boundaries=bounds )  ## boundaries表示边界值   
    # 标注 colorbar 的标签
    cbar.set_ticklabels(codelabels, fontsize=15) 

    axs[0].set_ylabel('samples')

    plt.tight_layout()
    #plt.savefig(savename+'.jpg', dpi=300, bbox_inches='tight')
    plt.show()






def plot_2dprofile_lithwell(int_pred, lithcode='-101',
                            savename='profile', section_name=None,
                            plot_section='inline',
                            pred_true_well=None, codelabel=None, p=None,
                            well_width=5, half_window=None):

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(7*2, 8), dpi=300)
    if section_name is not None:
        fig.suptitle(section_name, fontsize=20)
    print('注意岩性编码是', lithcode)

    ##砂岩红色，泥岩绿色，灰岩黄色
    if lithcode == '012':
        colors = ['green', 'red', 'yellow']
        codeticks = [0, 1, 2]
        bounds = [-0.5, 0.5, 1.5, 2.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    elif lithcode == '123':
        colors = ['green', 'red', 'yellow']
        codeticks = [1, 2, 3]
        bounds = [0.5, 1.5, 2.5, 3.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    elif lithcode == '-101':
        colors = ['yellow', 'green', 'red']
        codeticks = [-1, 0, 1]
        bounds = [-1.5, -0.5, 0.5, 1.5]
        codelabels = ['limestone', 'mudstone', 'sandstone']

    if codelabel is not None:
        codelabels= codelabel

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))

    pred_cube = int_pred.copy()
    pred_cube = pred_cube.transpose()

    ix, xx, tx = inline_xline_twt_to_grid(inline=pred_true_well[:, 0], 
                                          inline_sta=p['inline_sta'], 
                             xline=pred_true_well[:, 1], xline_sta= p['xline_sta'], 
                             twt=pred_true_well[:, 2], twt_sta=p['twt_sta'],
                             id=p['id'], xd=p['xd'], td=p['td'])
    sta = tx[0]
    endp=tx[-1]+1
    inline = ix[0]
    xline = xx[0]
    twt = tx[0]
    
    if plot_section == 'inline':
        print('绘制 inline 剖面！')
        pred_sec = pred_cube[:, :, inline]
        line = xx[0]
    elif plot_section == 'xline':
        pred_sec = pred_cube[:, xline, :]
        line = ix[0]
        print('绘制 xline 剖面！')
    else:
        pred_sec = pred_cube[twt, :, :]
        print('绘制 depth 剖面！')
    #lith = pred_sec[sta:endp, line]
    lith = pred_true_well[:, 3]
    print('注意使用井预测文件的第3列的数据是真实岩性(注意甄别XX列)，以及减去半样本长度')


    im1 = ax.imshow(pred_sec, cmap=cmap, norm=norm, 
                    interpolation='nearest', aspect='auto')
    ax.set_xlabel('pred lith', fontsize=20)

    ##使用 extend='both' 映射外溢的标签
    cbar = plt.colorbar(im1,  # extend='both', 
                        ticks=codeticks,  # ticks指定了刻度
                        boundaries=bounds)  ## boundaries表示边界值   
    # 标注 colorbar 的标签
    cbar.set_ticklabels(codelabels, fontsize=15)

    ax.set_ylabel('samples')

    

    for i in range(-well_width, well_width):
        pred_sec[sta-half_window:endp-half_window, line+i] = pd.Series(lith).values

    im2 = ax.imshow(pred_sec, cmap=cmap,  norm=norm,
                interpolation='nearest', aspect='auto',
               origin='upper', alpha=1)

    plt.tight_layout()
    # plt.savefig(savename+'.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def jupyter_cigvis_3d_and_surfaces(data, sf1, sf2, cmin=None, cmax=None, 
                                   cmap='Petrel', cbar_label='rgt', 
              cbar=True, save_dir=None, save_name=None):
    if (cmin != None) & (cmax != None):
        cmin=cmin
        cmax=cmax
    else:
        cmin=np.nanmin(data)
        cmax=np.nanmax(data)
        print('自动除却nan后数据的cmin, cmax', cmin, cmax)

    nodes3 = cigvis.create_slices(data, cmap=cmap, clim=[cmin, cmax],
                    show_cbar=cbar)  #  cbar
    # show amplitude
    nodes3 += cigvis.create_surfaces([sf1, sf2],
                                    volume=data,
                                    value_type='amp',
                                    cmap=cmap,
                                    clim=[data.min(), data.max()])
    
    cigvis.plot3D(nodes3, size=(800, 800),
        savedir= save_dir,
                savename=save_name)


def jupyter_cigvis_3d(data, cmin=None, cmax=None, cmap='Petrel', 
                      cbar_label='rgt', 
              cbar=True, save_dir=None, save_name=None):
    """使用cigvis可视化三维数据"""

    if (cmin != None) & (cmax != None):
        cmin=cmin
        cmax=cmax
    else:
        cmin=np.nanmin(data)
        cmax=np.nanmax(data)
        print('自动除却nan后数据的cmin, cmax', cmin, cmax)

    nodes3 = cigvis.create_slices(data, cmap=cmap, clim=[cmin, cmax],
                    show_cbar=cbar)  #  cbar


    cigvis.plot3D(nodes3, size=(800, 800),
        savedir= save_dir,
                savename=save_name)


def visplot_3d(data, cmin=None, cmax=None, cmap='Petrel', 
                      cbar_label='rgt', nancolor=None,
              cbar=True, save_dir=None, save_name=None):
    """使用cigvis可视化三维数据"""

    if (cmin != None) & (cmax != None):
        cmin=cmin
        cmax=cmax
    else:
        cmin=np.nanmin(data)
        cmax=np.nanmax(data)
        print('自动除却nan后数据的cmin, cmax', cmin, cmax)

    nodes3 = viserplot.create_slices(data, cmap=cmap, clim=[cmin, cmax], nancolor = nancolor)  #  cbar #show_cbar=cbar
    ##记住viserplot暂时只有三个函数可以用，还没有办法同时绘制多个三维体
    # 仅支持 create_slices, add_mask(RGT fault), create_surfaces(层位) 这三个创建函数


    viserplot.plot3D(nodes3, size=(800, 800),
        savedir= save_dir,
                savename=save_name)
    

def visplot_3d_V2(data3d, server=8000):
    """"""

    server = viserplot.create_server(server)
    nodes = viserplot.create_slices(data3d) 
    
    viserplot.plot3D(nodes, server=server)  #logs_type='line', width=3#测井使用line线显示







def plot_input_imshow_label_V2(inputxlabel=None, seis=None, far=None, mid=None, near=None,
                            interp=None, rgt=None, phase=None, 
                         freq=None, ampli=None, imp=None, spec=None, log=None,
                         plot_label=False, label_over=False, label_type='imshow',
                         lithcode='-101', seis_depth=None, log_depth=None,
                         well_name=None, codelabel=None, savename=None,
                         plot_spec=True):
    """
    绘制完整的抽取的trace,包括二维的频谱
    以及label

    num: 输入数据类型的数目
    """

    ##砂岩红色，泥岩绿色，灰岩黄色
    if lithcode=='012':
        colors=['green', 'red', 'yellow']
        codeticks = [0,1,2]
        bounds = [-0.5, 0.5, 1.5, 2.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    if lithcode=='123':
        colors=['green', 'red', 'yellow']
        codeticks = [1,2,3]
        bounds = [0.5, 1.5, 2.5, 3.5]
        codelabels = ['mudstone', 'sandstone', 'limestone']
    if lithcode=='-101':
        colors=['yellow', 'green', 'red']
        codeticks = [-1,0,1]
        bounds = [-1.5, -0.5, 0.5, 1.5]
        codelabels = ['limestone', 'mudstone', 'sandstone']
    if lithcode == '01234567':
        colors = ['green', 'red', 'yellow', 'blue',  'purple', 
                  'cyan', 'orange', 'brown']
        codeticks = [0, 1, 2, 3, 4, 5, 6, 7]
        bounds = [-0.5 + i for i in range(9)]  # [-0.5, 0.5, ..., 6.5]
        codelabels = ['0', '1', '2', '3', '4', '5', '6', '7']


    if codelabel is not None:
        codelabels= codelabel

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))

    log2d = log[:, np.newaxis]

    #inputxlabel = [seis, interp, rgt, phase, freq, ampli, imp, spec]
    ori_data = [seis, far, mid, near, interp, rgt, phase, freq, ampli, imp, spec]
    data = [x for x in ori_data if x is not None]

    ##判断是否需要绘制label
    if plot_label == True:
        nums = len(inputxlabel)+1
    else:
        nums = len(inputxlabel)

    fig, ax = plt.subplots(1, nums, figsize=(nums*3,10), dpi=300, sharey = True)
    if well_name is not None:
        fig.suptitle(str(well_name), fontsize=20)
    #y = np.arange(len(seis))
    

    if plot_label == False:
        for i in range(nums-1):
        ##这里减一是为了能够留出最后一列画频谱图
            ax[i].plot(data[i], seis_depth, color='b')  
            #ax[i].set_ylim(len(seis), 0)
            ax[i].set_xlabel(inputxlabel[i], fontsize=20)
            #ax[0].tick_params(axis='x', labelrotation=45)

        ax_1 = ax[-1].imshow(spec, cmap='jet', aspect='auto',
                             extent=[0, spec.shape[1], seis_depth[-1], seis_depth[0]])
        ax[-1].set_xlabel(inputxlabel[-1], fontsize=20)
        fig.colorbar(ax_1, ax = ax[-1])    

    else:
        if plot_spec:
            ax_1 = ax[-2].imshow(spec, cmap='jet', aspect='auto',
                             extent=[0, spec.shape[1], seis_depth[-1], seis_depth[0]])
            ax[-2].set_xlabel(inputxlabel[-1], fontsize=20)
            fig.colorbar(ax_1, ax = ax[-2])  
            nums_1 =  nums-2
        else:
            nums_1 = nums-1

        for i in range(nums_1):
        ##这里减一是为了能够留出最后一列画频谱图
            if label_over==True:
                ##当label_over==True时，每列子图label和输入曲线画在一起
                ax[i].imshow(log2d, cmap=cmap, norm=norm, 
                             interpolation='nearest', aspect='auto',
                             extent=[0, log2d.shape[1], log_depth[-1], log_depth[0]]
                             )  #extent=[data[i].min(), data[i].max(), y.min(), y.max()]
                #print('每列子图label和输入曲线画在一起')   
            ax[i].plot(data[i], seis_depth, color='b', linewidth=5) 
            
            #ax[i].set_ylim(len(seis), 0)
            ax[i].set_xlabel(inputxlabel[i], fontsize=20)
            #ax[0].tick_params(axis='x', labelrotation=45)
        
        if label_type=='imshow':
            im = ax[-1].imshow(log2d, cmap=cmap, norm=norm, 
                            interpolation='nearest', aspect='auto',
                            extent=[0, log2d.shape[1], log_depth[-1], log_depth[0]]  #[left, right, bottom, top]
                            )
            ##使用 extend='both' 映射外溢的标签
            cbar = plt.colorbar(im, #extend='both', 
                                ticks=codeticks, #ticks指定了刻度
                                boundaries=bounds )  ## boundaries表示边界值   
            # 标注 colorbar 的标签
            #print('11111')
            cbar.set_ticklabels(codelabels, fontsize=15) 


        elif label_type == 'line':
            ax[-1].plot(log, log_depth, color='r')  
            # ax[-1].set_ylim(len(log), 0)
            # ax[-1].set_xlabel('true lith', fontsize=20)
        #ax[-1].set_ylim(len(log)-1, 0)
        ax[-1].set_xlabel('true lith', fontsize=20)

    plt.tight_layout() 

    if savename:
        plt.savefig(savename+'.jpg', dpi=300, bbox_inches='tight')

    plt.show()





def visualize_maskc(maskc,well,name, plot_len, save_path):
    #可视化maskc或其一部分
    """
    plot_len:截取一定长度的数据来绘制maskc
    """

    fig,axes = plt.subplots(1,2,sharey=True,figsize=(5,10),dpi=300)   #(wigth,height)
    if plot_len is None:
        plot_len = len(maskc)
        print("======== plot_len（mask绘制的长度）", plot_len)
    else:
        plot_len = plot_len  #plot_len是为查看多少个采样点  绘制80000个采样点的mask的情况
    msec = maskc[0:plot_len].flatten()    #ndarray.flatten()展平数组，C 按行展平（默认）；F 按列展平

    wellsec = pd.DataFrame(well.iloc[0:plot_len].copy(),columns=['WELL'])  #对于datafraame，columns要传入集合形式，如['WELL']可以，而‘WELL’则不可
    #print('wellsec.columns',wellsec.columns,wellsec.shape)
    base = 11111
    well_name = wellsec['WELL'].unique()
    for it in range(len(well_name)):
        wellsec[wellsec['WELL']==well_name[it]] = base*(it+1)

    #print('wellsec',wellsec)
    y = [i for i in range(len(msec))]
    axes[0].imshow(np.array(wellsec,dtype=np.float64),interpolation='nearest',aspect='auto')  #imshow中的数据为数组时，数据需为浮点型
    axes[1].plot(msec,y,color='k')

    plt.tight_layout()
    plt.savefig(save_path+str(name)+'maskc.png')



def visualize_logs_preds_targets(fea_oris, preds, targets, lith_code_map_name,
                                 save=None, wname=None, path=None):
    """
    fea_oris: (n, m)  每列为一种曲线
    preds: (n,)      预测标签，取值 0-11
    targets: (n,)    真实标签，取值 0-11
    lith_code_map_name: dict {原始编码: 名称}，下述会按key升序排列
    """
    n, m = fea_oris.shape
    lith_type_num = len(lith_code_map_name)
    # 定义一个 colormap，12 个类别

    ### V1: 调色板根据岩性种类数目来变化
    #cmap = plt.get_cmap("tab20", lith_type_num)  # tab20 可分多色，这里取 12 种


    ### v2:固定全局12色调色板
    print("====== v2:岩性colormap固定全局12色调色板")
    base_cmap = plt.get_cmap("tab20", 12)  
    # 取前 lith_type_num 个颜色（保持原顺序一致）
    colors = [base_cmap(i) for i in range(lith_type_num)]
    # 自定义 colormap（从固定色中截取前 n 种）
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)


    
    fig, axes = plt.subplots(1, m+2, figsize=(3*(m+2), 13), sharey=True, dpi=300)
    plt.subplots_adjust(wspace=0.1)
    
    # 1) 绘制每个测井曲线
    for j in range(m):
        axes[j].plot(fea_oris[:, j], np.arange(n), color='black')
        axes[j].set_title(f"Curve {j+1}", fontsize=15)
        axes[j].invert_yaxis()  # 深度方向朝下更符合习惯
        axes[j].grid(True, linestyle='--', alpha=0.3)
        if j == 0:
            axes[j].set_ylabel("Depth")

    # 2) 绘制预测结果
    im1 = axes[m].imshow(preds[:, None], aspect='auto', cmap=cmap,
                         vmin=0, vmax=lith_type_num-1,
                         interpolation='nearest')
    axes[m].set_title("Preds", fontsize=15)
    axes[m].set_xticks([])
    axes[m].set_yticks([])

    # 3) 绘制真实结果
    im2 = axes[m+1].imshow(targets[:, None], aspect='auto', cmap=cmap,
                           vmin=0, vmax=lith_type_num-1,
                           interpolation='nearest')
    axes[m+1].set_title("Labels", fontsize=15)
    axes[m+1].set_xticks([])
    axes[m+1].set_yticks([])

    # 颜色条
    # 颜色条：按 key 排序获取岩性名称
    sorted_items = sorted(lith_code_map_name.items(), key=lambda x: x[0])
    lith_names = [v for _, v in sorted_items]

    # ax=axes → colorbar 与所有子图对齐，而不是只跟最后一个子图对齐。
    # fraction → 控制 colorbar 厚度（越小越细）。
    # pad → 控制 colorbar 与子图之间的间距。
    # aspect → 控制 colorbar 的纵横比（值越大，colorbar 越细长）。
    cbar = fig.colorbar(im2, ax=axes[m+1],orientation="vertical",   #ax=
                        fraction=0.046, pad=0.04, aspect=60)
    cbar.set_ticks(np.arange(lith_type_num))
    lith_names_wrapped = [name.replace(" ", "\n") for name in lith_names]
    cbar.set_ticklabels(lith_names_wrapped, fontsize=15)
    #cbar.set_ticklabels(lith_names)
    cbar.set_label("Lithology Class", fontsize=15)

    plt.suptitle(wname, fontsize=20)

    plt.tight_layout()
    if save is not None and path is None:
        plt.savefig(save + str(wname) +'.jpg', bbox_inches='tight',)
    elif save is not None and path is not None:
        plt.savefig(path + save + str(wname) +'.jpg', bbox_inches='tight',)
    plt.show()


def visualize_completion(fea_ori, fea_masked, fea_recon, masked_channels,
                         save_dir=None, wname=None):
    """
    可视化单口井（单条样本）的掩码重建结果。
    fea_ori, fea_masked, fea_recon: numpy数组或torch张量, shape = (L, D)
        L: 深度/采样点数量
        D: 测井曲线数量
    masked_channels: int或list[int], 掩码并重建的通道编号
    """
    # ---- 转换为 numpy ----
    if hasattr(fea_ori, "detach"):
        fea_ori = fea_ori.detach().cpu().numpy()
    if hasattr(fea_masked, "detach"):
        fea_masked = fea_masked.detach().cpu().numpy()
    if hasattr(fea_recon, "detach"):
        fea_recon = fea_recon.detach().cpu().numpy()

    # ---- 确保 masked_channels 是列表 ----
    if isinstance(masked_channels, int):
        masked_channels = [masked_channels]

    os.makedirs(save_dir, exist_ok=True)
    L, D = fea_ori.shape

    # ---- 绘制每个通道 ----
    fig, axes = plt.subplots(1, D, figsize=(3*D, 12), sharey=True, dpi=200)
    if D == 1:
        axes = [axes]

    for j in range(D):
        axes[j].plot(fea_ori[:, j], np.arange(L), label='orig', linewidth=1.2)
        axes[j].plot(fea_masked[:, j], np.arange(L), label='masked', linestyle='--', alpha=0.6)
        if j in masked_channels:
            axes[j].plot(fea_recon[:, j], np.arange(L), label='recon', linewidth=1.5)
            axes[j].set_title(f"Ch {j} (masked & recon)")
        else:
            axes[j].set_title(f"Ch {j}")

        axes[j].invert_yaxis()
        axes[j].grid(True, linestyle='--', alpha=0.3)
        if j == 0:
            axes[j].set_ylabel("Depth")
        axes[j].legend(loc='upper right', fontsize=8)

    plt.suptitle(f"{wname} (orig vs masked vs recon)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir is not None:
        fname = os.path.join(save_dir, f"{wname}_recon.jpg")
        plt.savefig(fname, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close(fig)




def plot_bar(values, labels, ylabel, title, save_path):
    # num_bars = len(values)

    # # 根据数量决定图宽度（每个条形 0.6 ～ 0.8 英寸）
    # width_per_bar = 0.6
    # fig_width = min(8, num_bars * width_per_bar)
    # plt.figure(figsize=(fig_width, 5))   # 自动宽度

    plt.figure(figsize=(10, 5)) 
    bars = plt.bar(labels, values)

    # 在柱子顶部添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,   # x 坐标
            height,                              # y 坐标
            f"{value:.3f}",                       # 显示格式（自动保留三位小数）
            ha='center', va='bottom', fontsize=12
        )

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"====== 已保存: {save_path}")


def save_confusion_matrix_heatmap(
    cm,
    class_labels,
    save_path,
    name="confusion_matrix",
    title=None,
    *,
    # 1) 统一颜色范围（方便对比）
    vmin=0.0, vmax=1.0,
    # 2) 类名换行
    wrap_labels_on_space=True,
    # 其他可调参数
    figsize=(10, 8), dpi=300, cmap="Blues",
    fmt=".3f", annot=True,
    annot_fontsize=22, label_fontsize=17, tick_fontsize=15,
    x_rotation=0, y_rotation=90,
    cbar=True, cbar_ticks=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
):
    """
    cm: 2D array-like, 归一化混淆矩阵（建议值域[0,1]）
    class_labels: list[str]
    """

    # --- 换行标签：遇到空格 -> '\n'
    if wrap_labels_on_space:
        tick_labels = [str(s).replace(" ", "\n") for s in class_labels]
    else:
        tick_labels = class_labels

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cmap=cmap,
        vmin=vmin,           # ✅ 固定色域下限
        vmax=vmax,           # ✅ 固定色域上限
        annot_kws={"fontsize": annot_fontsize},
        cbar=cbar
    )

    # 坐标轴标签
    ax.set_xlabel("Predicted", fontsize=label_fontsize)
    ax.set_ylabel("Ground Truth", fontsize=label_fontsize)

    # tick 字体 + 旋转
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    plt.xticks(rotation=x_rotation, ha="center")
    
    plt.yticks(rotation=y_rotation, ) #ha="right"
    



    # colorbar tick 固定 0~1（可选）
    if cbar and cbar_ticks is not None:
        cbar_obj = ax.collections[0].colorbar
        cbar_obj.set_ticks(list(cbar_ticks))
        cbar_obj.ax.tick_params(labelsize=tick_fontsize)

    if title is not None:
        plt.title(title, fontsize=label_fontsize + 2)

    plt.tight_layout()
    fpath = save_path + f"{name}.jpg"
    plt.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"====== 混淆矩阵热力图已保存到 {fpath}")

def plot_heatmap(matrix, wells, classes, title, save_path):
    plt.figure(figsize=(10,8))
    sns.heatmap(matrix, annot=True, fmt=".2f",
                xticklabels=classes,
                yticklabels=wells,
                cmap="YlGnBu", annot_kws={"fontsize": 12})
    # 设置横纵坐标字体
    plt.xticks(fontsize=13, )
    plt.yticks(fontsize=13)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"====== 已保存: {save_path}")


