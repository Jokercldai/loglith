import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """可学习位置编码（比固定sinusoidal更适合连续测井数据）"""
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]


class LogTransformer(nn.Module):
    """
    端到端 Transformer，用测井曲线分类岩性（无预训练）。
    输入:  (B, T, C) 连续测井曲线
    输出:  (B, T, num_classes)
    """
    def __init__(self, args,
                 d_model=128, nhead=4, num_layers=2, dropout=0.2,
                 max_len=2048):
        super().__init__()

        num_features = len(args.fea)
        num_classes = len(args.lith_code_map_name.keys())

        # 1) 输入投影层： C → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # 2) 可学习位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # 3) Transformer Encoder 堆叠
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True   # 重要：使输入为 (B, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) 分类头，逐点分类
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """统一初始化，提升训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, model_task=None, attention_mask=None):
        """
        x: (B, T, C)
        attention_mask: (B, T), 1=有效，0=padding
        """
        # 1) 输入投影
        x = self.input_proj(x)  # (B, T, d_model)

        # 2) 加位置编码
        x = self.pos_encoder(x)

        # 3) Transformer 编码
        out = self.encoder(x, src_key_padding_mask=(attention_mask == 0) 
                           if attention_mask is not None else None)

        # 4) 分类
        logits = self.classifier(out)  # (B, T, num_classes)

        return logits






###################  ResNet架构  ####################################
# class BasicBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm1d(out_channels)
#             )

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         return self.relu(out)

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        # dilation=1：局部模式
        # dilation=2：4倍感受野
        # dilation=4：8倍感受野

        padding = dilation  # 保持长度不变

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=3, padding=padding, stride=stride, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=3, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # shortcut 不改变长度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


# class LogResNet(nn.Module):
#     def __init__(self, args, base=64):
#         super().__init__()

#         num_features = len(args.fea)
#         num_classes = len(args.lith_code_map_name.keys())

#         self.layer1 = BasicBlock1D(num_features, base)
#         ##self.layer2 = BasicBlock1D(base, base*2, stride=2)
#         self.layer2 = BasicBlock1D(base, base*2, stride=1, dilation=2)
#         self.layer3 = BasicBlock1D(base*2, base*2)
#         ##self.layer4 = BasicBlock1D(base*2, base*4, stride=2)
#         self.layer4 = BasicBlock1D(base*2, base*4, stride=1, dilation=4)

#         self.classifier = nn.Sequential(
#             nn.Linear(base*4, base*2),
#             nn.ReLU(),
#             nn.Linear(base*2, num_classes)
#         )

#     def forward(self, x, model_task=None, ):
#         x = x.transpose(1, 2)  # (B,T,C)→(B,C,T)
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)   # (B,base*4,T)

#         out = out.transpose(1, 2)
#         logits = self.classifier(out)
#         return logits


class LogResNet(nn.Module):
    def __init__(self, args, base=64):
        super().__init__()
        print(f"====== 该 {args.model_type} 为简化版 ======")

        num_features = len(args.fea)
        num_classes = len(args.lith_code_map_name.keys())

        self.layer1 = BasicBlock1D(num_features, base)
        ##self.layer2 = BasicBlock1D(base, base*2, stride=2)
        self.layer2 = BasicBlock1D(base, base*2, stride=1, dilation=2)
        self.layer3 = BasicBlock1D(base*2, base*2)
        ##self.layer4 = BasicBlock1D(base*2, base*4, stride=2)
        self.layer4 = BasicBlock1D(base*2, base*4, stride=1, dilation=4)

        # self.classifier = nn.Sequential(
        #     nn.Linear(base*4, base*2),
        #     nn.ReLU(),
        #     nn.Linear(base*2, num_classes)
        # )
        ##简化版
        self.classifier = nn.Sequential(
            nn.Linear(base*2, num_classes)
        )

    def forward(self, x, model_task=None, ):
        x = x.transpose(1, 2)  # (B,T,C)→(B,C,T)
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)   # (B,base*4,T)

        out = out.transpose(1, 2)
        logits = self.classifier(out)
        return logits




###################  CNN-BiLSTM架构  ####################################
# class CNNBiLSTM(nn.Module):
#     def __init__(self, args,
#                  cnn_channels=64, lstm_hidden=128, num_layers=2, bidirectional=True):
#         super().__init__()

#         print(f"====== 使用 CNN-BiLSTM 模型架构！！！是否双向: {bidirectional} ======")

#         num_features = len(args.fea)
#         num_classes = len(args.lith_code_map_name.keys())

#         # CNN 提取局部模式
#         self.conv = nn.Sequential(
#             nn.Conv1d(num_features, cnn_channels, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
#             nn.ReLU()
#         )

#         # BiLSTM 学习序列依赖
#         self.lstm = nn.LSTM(
#             cnn_channels, lstm_hidden,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional #True
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(lstm_hidden * 2 if bidirectional else lstm_hidden, lstm_hidden),
#             nn.ReLU(),
#             nn.Linear(lstm_hidden, num_classes)
#         )

#     def forward(self, x, model_task=None,):
#         x = x.transpose(1, 2)         # (B,C,T)
#         x = self.conv(x)
#         x = x.transpose(1, 2)         # → (B,T,C)
#         out, _ = self.lstm(x)
#         logits = self.classifier(out)
#         return logits



class CNNBiLSTM(nn.Module):
    def __init__(self, args,
                 cnn_channels=64, lstm_hidden=128, num_layers=2, bidirectional=True):
        super().__init__()

        print(f"====== 使用 CNN-BiLSTM 模型架构！！！是否双向: {bidirectional} ======")
        print(f"====== 该 {args.model_type} 为简化版 ======")

        num_features = len(args.fea)
        num_classes = len(args.lith_code_map_name.keys())

        # CNN 提取局部模式
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),

            ##简化掉，只留一层
            # nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            # nn.ReLU()
        )

        # BiLSTM 学习序列依赖
        self.lstm = nn.LSTM(
            cnn_channels, lstm_hidden,
            num_layers=num_layers,  #num_layers由2简化为1
            batch_first=True,
            bidirectional=bidirectional #True
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(lstm_hidden * 2 if bidirectional else lstm_hidden, lstm_hidden),
        #     nn.ReLU(),
        #     nn.Linear(lstm_hidden, num_classes)
        # )
        ##简化
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 if bidirectional else lstm_hidden, num_classes),
        )

    def forward(self, x, model_task=None,):
        x = x.transpose(1, 2)         # (B,C,T)
        x = self.conv(x)
        x = x.transpose(1, 2)         # → (B,T,C)
        out, _ = self.lstm(x)
        logits = self.classifier(out)
        return logits

