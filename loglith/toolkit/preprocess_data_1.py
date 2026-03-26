##制作目标层段外的三维mask
##归一化三维数据，
##获取各种岩性标签在三维体中的index
"""
mask_volume_by_horizon: 制作目标层位内的mask体
inline_xline_twt_to_grid将实际工区的inline xline twt转化为网格采样点上的
trans_twthor_to_gridhor 将层位数据的twt值调整为地震数据的网格值
lithseg_to_lithpoint(): 将井岩性柱/段转化为岩性采样点的形式
save_pred_lith 保存预测的井岩性和真实的井岩性(包括其inline xline twt)
gaussian_filter 对一维数据进行平滑
plot_test_gaussian_window 测试gaussian平滑的合适窗口，并绘图
bandpass_filter() 对seis进行分频分析
vis_seisfreq_wave() 可视化地震对应频带的波形
read_horizon_head_and_txt_file(file_path): 将层位的txt(含表头)提取成txt数据文件
resample_data_any_rate(data, rate=2)  按照任意倍率进行下采样，并且保留自定义的优先岩性
Statistical_sample_location  统计选择的样本的位置，data:二维数组，前三列表示inline xline twt 第四列为1用于统计
find_error_preds_with_depths:统计预测列和ground_truth列不对应的值对应的深度以及well名，存储为字典
save_dict_to_txt: 将字典保存为json,存储进txt中
load_dict_from_txt：
read_memmap_multi_3ddata:动态读取任意数量的3D数据文件，并返回与输入路径相对应的多个3D数组。
get_specs:获取多个频谱数据并拼接成一个二维数组。
"""

import pandas as pd 
import numpy as np
from stockwell import st
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from collections import defaultdict
import json







def read_memmap_multi_3ddata(*paths, shape, dtype=np.float32, mode='c'):
    """
    动态读取任意数量的3D数据文件，并返回与输入路径相对应的多个3D数组。
    
    参数：
    *paths (str): 任意数量的文件路径。
    shape (tuple): 3D数据的形状，例如 (n3, n2, n1)。
    dtype (type, optional): 数据类型，默认为 np.float32。
    mode (str, optional): 读取模式，默认为 'c'。
    
    返回：
    多个3D数组，数量与输入路径相同。
    """
    data_arrays = []
    
    for path in paths:
        try:
            data = np.memmap(path, shape=shape, dtype=dtype, mode=mode)
            data_arrays.append(data)
            #print(f"Successfully loaded data from {path}")
        except Exception as e:
            print(f"该路径数据加载错误 {path}: {e}")
            data_arrays.append(None)  # 如果读取失败，将该项设为 None
    return tuple(data_arrays)  # 返回一个元组，包含所有加载的3D数组



def get_specs(ix, xx, tx, *specs):
    """
    获取多个频谱数据并拼接成一个二维数组。
    
    参数：
    ix, xx, tx: 数据索引。
    *specs: 多个 3D 频谱数据数组。
    
    返回：
    去除 NaN 值后的拼接二维数组。
    """
    dsp_allx_ori = np.column_stack([spec[ix, xx, tx] for spec in specs])
    return dsp_allx_ori[~np.isnan(dsp_allx_ori)].reshape(-1, 
                                                         dsp_allx_ori.shape[1])


# 自定义 JSON 编码器  使得ndarray能够转为json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_dict_to_txt(dictionary, filename, mode='w'):
    """
    将字典存储到文件中

    :param dictionary: 要存储的字典
    :param filename: 文件名
    :param mode: 写入模式，'w' 为覆盖写，'a' 为添加写（默认 'w'）
    """
    with open(filename, mode) as file:
        json.dump(dictionary, file, cls=NumpyEncoder)
        file.write("\n")  # 添加换行符以便追加写入时区分不同字典

def load_dict_from_txt(filename):
    """
    从文件中加载字典
    :param filename: 文件名
    :return: 读取的字典
    """
    dictionary = {}
    with open(filename, 'r') as file:
        # 读取所有行
        lines = file.readlines()
        for line in lines:
            if line.strip():  # 如果行不为空, # 跳过空行
                # 解析 JSON 数据并更新字典
                dictionary.update(json.loads(line))
    
    return dictionary



def find_error_preds_with_depths(data, target_values, dict, wname, truth_col=3, pred_col=4):
    """
    查找第三列中为 target_values 且在第二列中的值不相等的位置对应的深度值
    :param data: 二维数组，形状为 (n, 3)
    :param target_values: 目标值列表，可以包含多个值，如 [1, 2, 3]
    :return: 满足条件的深度值列表
    默认深度列为0列，预测值为1列， 真值为2列
    """
    target_values = np.array(target_values)  # 将列表转换为数组
    condition = np.isin(data[:, truth_col], target_values) & (data[:, pred_col] != data[:, truth_col])
    #print("预测不正确的数目：", data[condition, :].shape[0])
    dict[wname] = data[condition, :]
    return dict


def Statistical_sample_location(data2d):
    """统计选择的样本的位置，
    data:二维数组，前三列表示inline xline twt 第四列为1用于统计"""
    #使用一个字典来统计每一组前三列相同的行的第四列的和。
    # #将字典中的数据转换为一个列表，并按照要求排序。
    # # Step 1: Use a dictionary to sum the fourth column for rows with the same first three columns
    summary_dict = defaultdict(int)
    for row in data2d:
        key = tuple(row[:3])
        summary_dict[key] += row[3]

    # Step 2: Convert the dictionary to a sorted list
    result = [[*key, value] for key, value in summary_dict.items()]
    result.sort()
    return result


def resample_data_any_rate(data, rate=2, first_resam_lith=1, all_prior=[1, 2, 3, 4, 5, 6, 7],
                           first_prior=[1, 2, 3, 4, 5], trans_first_resam_lith=True,
                           trans_to_0=True):
    """按照任意倍率进行下采样，并且保留自定义的优先岩性
    first_resam_lith表示保留的数据重新赋值"""

    depth_col = data[:, 0]
    lithology_col = data[:, 1]

    # 重采样时间列
    depth_resampled = depth_col[::rate]
    #print('depth_resampled应该全是整数:', depth_resampled)

    # 创建新的岩性列
    lithology_resampled = []
    for i in range(0, len(lithology_col), rate):
        # 当前采样窗口
        window = lithology_col[i:i + rate]
        
        # 如果窗口内有1-7的值，优先保留
        if any(value in window for value in all_prior):
            for priority_value in first_prior:
                if priority_value in window:
                    if trans_first_resam_lith:
                        lithology_resampled.append(first_resam_lith)
                    else:
                        lithology_resampled.append(priority_value)
                    break
                    
            else:
                #print("np.nonzero(window), window[np.nonzero(window)]", np.nonzero(window), window[np.nonzero(window)]) #(array([0, 1]),) [6. 6.]
                #print('window[np.nonzero(window)][0]', window[np.nonzero(window)][0])
                lithology_resampled.append(window[np.nonzero(window)][0])
        else:
            lithology_resampled.append(0) 
            # if trans_to_0:
            #     lithology_resampled.append(0)  #这个0就是多数岩性的编码
            # else:
            #     lithology_resampled.append(int(np.mean(window[1:-1]))) 


    # 将结果转换为numpy数组并返回
    return np.column_stack((depth_resampled, lithology_resampled))


def delete_incomplete_head_and_tail(data, od, interval=1):
	"""
	由于致密的岩性其首尾会有不完整的岩性，在匹配到地震网格上会错误，所以删去这一首尾部分
    od表示深度列序号
	"""
	# 查找第三列第一个小数部分为0的数据的行索引A
	A = np.where((data[:, od] % interval) == 0)[0][0]
	# 查找第三列最后一个小数部分为0的数据的行索引B
	B = np.where((data[:, od] % interval) == 0)[0][-1]
	# 取出A和B之间的数组
	result = data[A:B]  ##不取最后一个为0的
	return result


def read_horizon_head_and_txt_file(file_path):
    """将txt文件的表头和列数据分别读取出来,返回txt前的表头，txt列数据，和层位深度数据"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        print(f"原始数据的行数len(lines): {len(lines)}")
    
    # 分离表头和数据部分
    header_lines = []
    data_lines = []
    is_data = False
    for line in lines:
        if line.startswith("#") or not is_data:
            header_lines.append(line.strip())
            if line.strip() == "# End:":
                is_data = True
        else:
            data_lines.append(line.strip())
    
    # 将表头部分读取为字典
    header = {}
    for line in header_lines:
        if line.startswith("#"):
            key_value = line[1:].strip().split(":", 1)
            if len(key_value) == 2:
                key, value = key_value
                header[key.strip()] = value.strip()
    
    # 将数据部分读取为DataFrame，之后再转化为numpy
    data = []
    for line in data_lines:
        data.append(line.split())
    
    columns = ["x", "y", "twt ms", "Inline", "Xline"]
    data_df = pd.DataFrame(data, columns=columns)
    data_df = data_df.astype(float)
    data_np = data_df[["Inline", "Xline", "twt ms"]].values
    twt = data_np[:,-1]
    
    return header, data_np, twt

def gaussian_filter(data, sigma=4, ifenve=False):

    if ifenve == True:
        ##换成高斯滤波器对地震数据的最后一个维度滤波
        analytic_signal = hilbert(data)
        data = np.abs(analytic_signal)

    smoothed = gaussian_filter1d(data, sigma=sigma)

    return smoothed


# def gaussian_smoothing_selfdesign_v1(signal, sigma):
#     """
#     Apply Gaussian smoothing to a sparse pulse wave signal.
    
#     Parameters:
#     - signal: 1D array-like, the input sparse pulse wave signal
#     - sigma: float, standard deviation for Gaussian kernel
    
#     Returns:
#     - smoothed_signal: 1D array-like, the smoothed signal
#     """
#     # Compute the Gaussian kernel
#     kernel_radius = int(3 * sigma)
#     x = np.arange(-kernel_radius, kernel_radius + 1)
#     gaussian_kernel = np.exp(-(x**2) / (2 * sigma**2))
#     gaussian_kernel /= gaussian_kernel.sum()  # Normalize kernel
    
#     # Perform convolution
#     smoothed_signal = np.convolve(signal, gaussian_kernel, mode='same')
    
#     # Find local maxima in the original signal
#     original_peaks = np.where(signal > 0)[0]
    
#     # Normalize to keep the original amplitude at each pulse
#     for peak in original_peaks:
#         peak_region = smoothed_signal[max(0, peak - kernel_radius): min(len(signal), peak + kernel_radius + 1)]
#         if np.max(peak_region) != 0:
#             amplitude_ratio = signal[peak] / np.max(peak_region)
#             smoothed_signal[max(0, peak - kernel_radius): min(len(signal), peak + kernel_radius + 1)] *= amplitude_ratio
    
#     return smoothed_signal



def gaussian_kernel(sigma, truncate=3.0):
    """
    Create a Gaussian kernel given a sigma and a truncate range.
    
    Parameters:
    - sigma: float, standard deviation of the Gaussian kernel.
    - truncate: float, the radius of the kernel will be truncate * sigma.
    
    Returns:
    - kernel: 1D numpy array, the Gaussian kernel.
    """
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def gaussian_smoothing_selfdesign(signal, sigma):
    """
    Apply Gaussian smoothing to a sparse pulse wave signal, preserving the amplitude.
    
    Parameters:
    - signal: 1D array-like, the input sparse pulse wave signal.
    - sigma: float, standard deviation for Gaussian kernel.
    
    Returns:
    - smoothed_signal: 1D array-like, the smoothed signal with preserved amplitudes.
    """
    # Compute the Gaussian kernel
    kernel = gaussian_kernel(sigma)
    kernel_radius = len(kernel) // 2
    
    # Perform convolution
    smoothed_signal = np.convolve(signal, kernel, mode='same')
    
    # Normalize to keep the original amplitude at each pulse
    for i in range(len(signal)):
        if signal[i] > 0:
            local_region = smoothed_signal[max(0, i - kernel_radius): min(len(signal), i + kernel_radius + 1)]
            if np.max(local_region) != 0:
                amplitude_ratio = signal[i] / np.max(local_region)
                smoothed_signal[max(0, i - kernel_radius): min(len(signal), i + kernel_radius + 1)] *= amplitude_ratio
    
    return smoothed_signal





def adaptive_gaussian_smoothing(signal):
    """
    Apply adaptive Gaussian smoothing to a sparse pulse wave signal,
    preserving the amplitude of pulses.

    Parameters:
    - signal: 1D array-like, the input sparse pulse wave signal.

    Returns:
    - smoothed_signal: 1D array-like, the smoothed signal with preserved amplitudes.
    """

    def adaptive_sigma(i, signal):
        # Calculate an adaptive sigma based on local pulse density
        window_size = 300  # Adjust as needed
        start = max(0, i - window_size)
        end = min(len(signal), i + window_size + 1)
        local_density = np.sum(signal[start:end] > 0)
        return max(1, 0.3 * local_density)  # Adjust multiplier for appropriate scaling

    def gaussian_kernel(sigma, truncate=3.0):
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel

    # Initialize smoothed signal
    smoothed_signal = np.zeros_like(signal, dtype=float)

    # Traverse signal to apply adaptive smoothing
    for i in range(len(signal)):
        if signal[i] > 0:
            sigma = adaptive_sigma(i, signal)
            kernel = gaussian_kernel(sigma)
            kernel_radius = len(kernel) // 2
            start = max(0, i - kernel_radius)
            end = min(len(signal), i + kernel_radius + 1)

            # Apply convolution with adaptive kernel
            smoothed_signal[start:end] += signal[i] * kernel[:end-start]

    # # Normalize around pulse positions to maintain amplitude 1
    # for i in range(len(signal)):
    #     if signal[i] > 0:
    #         smoothed_signal[i] /= smoothed_signal[i]
    # Normalize around pulse positions to maintain amplitude 1
    for i in range(len(signal)):
        if signal[i] > 0:
            local_region = smoothed_signal[max(0, i - kernel_radius): min(len(signal), i + kernel_radius + 1)]
            if np.max(local_region) != 0:
                amplitude_ratio = signal[i] / np.max(local_region)
                smoothed_signal[max(0, i - kernel_radius): min(len(signal), i + kernel_radius + 1)] *= amplitude_ratio


    return smoothed_signal



def plot_test_gaussian_window(data, sigma_values, ifenve=False):

    ori_lith = data.copy()

    if ifenve == True:
        ##换成高斯滤波器对地震数据的最后一个维度滤波
        analytic_signal = hilbert(data)
        data = np.abs(analytic_signal)

    # 创建绘图窗口
    fig, axes = plt.subplots(len(sigma_values) + 2, 1, figsize=(10, 10), sharex=True)

    # 原始数据
    axes[0].plot(ori_lith, label='Original Data')
    axes[0].set_title('Original Data')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(data, label='Convert Data')
    axes[1].set_title('Convert Data')
    axes[1].legend()
    axes[1].grid(True)


    # 平滑数据
    for i, sigma in enumerate(sigma_values):
        smoothed = gaussian_filter1d(data, sigma=sigma)  #, mode='nearest'
        axes[i + 2].plot(smoothed, label=f'Smoothed Data (sigma={sigma})')
        axes[i + 2].set_title(f'Gaussian Smoothing with sigma={sigma}')
        axes[i + 2].legend()
        axes[i + 2].grid(True)

    # 调整布局
    plt.tight_layout()
    plt.show()



def save_pred_lith(true, pred, well_name, save_path):
    ##将真实井和pred井cat在一起
    true_pred = np.concatenate([true, pred[:,np.newaxis]], axis=1)
    df = pd.DataFrame(true_pred, columns=['inline', 'xline', 'twt', 'true', 'pred_lith'])
    
    df.to_csv(save_path+well_name+'.csv', index=False)



def inline_xline_twt_to_grid(inline=None, inline_sta=None, 
                             xline=None, xline_sta=None, 
                             twt=None, twt_sta=None,
                             id=None, xd=None, td=None):
     
    """
    将三维地震数据的inline xline twt原始道号转化为网格点号
    id, xd, td分别表示inline,xline,twt间隔
    """
    
    if inline is not None:
        new_i = (inline-inline_sta)/id
    if xline is not None:
        new_x = (xline-xline_sta)/xd
    if twt is not None:
        new_t = (twt-twt_sta)/td

    new_i = new_i.astype(int)
    new_x = new_x.astype(int)
    new_t = new_t.astype(int)

    return new_i, new_x, new_t
    

def mask_volume_by_horizon(seis, top_horizon, bottom_horizon):
    """
    三维数据顶底界面可能含有0值或者nan值等
    按照有效的top和bottom层位构建mask体，避免0值的干扰
    返回得mask_volume中，有效层位内对应的mask的值为1，其余的为nan
    """

    mask = np.zeros_like(seis)
    assign = np.arange(0, mask.shape[2])
    ##省略号表示未指定维度，可以在三维数组的每一个维度上广播赋值
    mask[..., :] = assign  
    #print('mask[:2,:2, :5]', mask[:2,:2, :5])
    maskA=((mask >= top_horizon[:, :, np.newaxis]) & 
           (mask <= bottom_horizon[:, :, np.newaxis]))
    mask[maskA] = 1
    mask[~maskA] = np.nan

    return mask 


    # ### 读取层位的数据，并循环判断层位内位置处madk=1
    # ##打印的T50，T70数据已转换为网格点，而不是深度
    # T50 = np.fromfile(p['T50'], dtype=np.float32).reshape((p['n3'], p['n2']))
    # T70 = np.fromfile(p['T70'], dtype=np.float32).reshape((p['n3'], p['n2']))
    # print('T50[:5, :5]', T50[:5, :5]) 
    
def Normalize_3D(data3d, mask):
    """
    均值方差归一化一个三维数据体，该数据体在top和bottom层位外有连续的0值或null值，
    归一化时按照top和bottom层位排除外界0值的干扰
    (mask是按照top和bottom构建的volume，界面外为nan值)
    返回的是归一化的三维体，界面之外的值用np.nan代替
    """

    valid3d = data3d*mask

    ##计算整个三维数组中所有非 NaN 值数据的均值，排除了nan值的干扰
    mean = np.nanmean(valid3d, axis=(0, 1, 2))
    std = np.nanstd(valid3d, axis=(0, 1, 2))
    print(f'数据体的均值为:{mean}，标准差为:{std}')

    normalized_data = (valid3d - mean) / std
    print('np.nanmax(), np.nanmin()', np.nanmax(normalized_data), np.nanmin(normalized_data))
    return normalized_data

def lith_index(lith_data, lithcode_1, lithcode_2, lithcode_3):
    """
    获取不同岩性的index，便于后续根据index来均匀选取岩性作为标签
    返回二维数组,维度(n,3)，n表示共有n个该类岩性，3中存储道号(inline, xline, depth)
    """

    lith1_index = np.argwhere(lith_data == lithcode_1)
    lith2_index = np.argwhere(lith_data == lithcode_2)
    lith3_index = np.argwhere(lith_data == lithcode_3)

    return lith1_index, lith2_index, lith3_index


def seis_diff_spec(trace, min_freq=6, max_freq=45):
    """该trace已经去除nan值，并且长度不小于85
    返回分类选取的样本对应的那一段差分频谱"""
    print('频谱最小频率:', min_freq)
    ###在loda_data中提取每个样本的频谱，该频谱不再进行归一化
    stock = st.st(trace, min_freq, max_freq, gamma=0.3) 
    spectra = np.abs(stock)
    spectra_T = spectra.T
    spectra_T = spectra_T.astype(np.float32)
    #print('spectra_T.shape', spectra_T.shape, spectra_T.dtype, spectra_T.sum())
    ###对频谱数据频率维度作差分
    spectra_T = np.diff(spectra_T, axis=0)
    ##由于差分少一行，所以vstack将最后一行复制，vstack会自动维度匹配
    spectra_T = np.vstack((spectra_T, spectra_T[-1, :]))
    #print('返回的差分频谱的shape,应是(sam_size,40)', spectra_T.shape, spectra_T.dtype)
    
    return spectra_T

def sepc_mean_by_section(sepc_shu, zone_num = 8):
	"""将频谱按照分8个频段求平均，方便后续作为8个通道"""
	# 定义每个区间的列范围
	inte=int(sepc_shu.shape[1]/zone_num)
	#print('频段间隔', inte)
	# 计算每个区间的起始和结束列索引
	start_cols = np.arange(0, sepc_shu.shape[1], inte)
	end_cols = start_cols + inte
	# 初始化结果数组
	result_sepc_shu = np.full((sepc_shu.shape[0], zone_num), np.nan)
	result_sepc_shu[:, :] = np.mean(np.split(sepc_shu, end_cols[:-1], axis=1), axis=-1).T
	result_sepc_shu = result_sepc_shu.astype(np.float32)
	return result_sepc_shu


## 将层位数据的twt值调整为地震数据的网格值trans_twthor_to_gridhor
def trans_twthor_to_gridhor(hor, twt_sta, td, save_file=False,save_dir=None,save_name=None):
    gridhor = (hor-twt_sta)/td
    gridhor = gridhor.astype(np.float32)
    if save_file==True:
        gridhor.tofile(save_dir+save_name)



# 设计带通滤波器
def bandpass_filter(seis_ori, lowcut, highcut, fs, order=5):
    seis = seis_ori.copy()
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, seis)
    return y

def vis_seisfreq_wave(seis, freq1, freq2, freq3, freq4, freq5, freq6, freq7, lith, 
                      freqband_num, well_name=None):
    num = freqband_num+2
    y = range(len(seis))
    fig,axs = plt.subplots(1,num, figsize=(num*2, 8), sharey=True, dpi=300)
    axs[0].plot(seis, y)
    axs[0].set_title('Original Seis')

    axs[1].plot(freq1, y)
    axs[1].set_title('(1-4Hz)')

    axs[2].plot(freq2, y)
    axs[2].set_title('5-14Hz')

    axs[3].plot(freq3, y)
    axs[3].set_title('15-24Hz')

    axs[4].plot(freq4, y)
    axs[4].set_title('25-34Hz')

    axs[5].plot(freq5, y)
    axs[5].set_title('35-44Hz')

    axs[6].plot(freq6, y)
    axs[6].set_title('45-64Hz')

    axs[7].plot(freq7, y)
    axs[7].set_title('65-84Hz')

    axs[8].plot(lith, y)
    axs[8].set_title('Lith')

    axs[0].set_ylabel('Sample')
    axs[0].invert_yaxis() #反转y轴

    if well_name is not None:
        fig.suptitle(f'{well_name}', fontsize=20)

    plt.tight_layout()
    plt.show()




    