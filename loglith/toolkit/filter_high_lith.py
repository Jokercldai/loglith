"""利用高频真实岩性正演获得的高分辨合成数据不变，继续利用该合成数据来作预训练。
在微调模型时，我们将真实岩性中单独的（仅一个点不连续）岩性赋值为其周围的背景值，
然后利用这经过过滤数据来继续微调训练预测。"""
"""所以我们在该文件中过滤单独的岩性值，赋值为周围的背景岩性"""

import numpy as np



    


def filter_hith_lith(trace_ori):  #save_path
    trace = trace_ori.copy()
    count = 0
    for i in range(1, len(trace) - 1):
        if trace[i] not in [trace[i-1], trace[i+1]]:
            trace[i] = trace[i-1]  #以左边的值作为背景值替代
    
            count+=1
            
    if trace[0] != trace[1]:
        trace[0] = trace[1]
        count+=1

    if trace[-1] != trace[-2]:
        trace[-1] = trace[-2]
        count+=1

    print(f"单独值个数{count}")
    return trace
