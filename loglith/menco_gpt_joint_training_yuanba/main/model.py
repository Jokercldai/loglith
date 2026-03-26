import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
import math


# ================================================================
# MaskedDecoder 保持不变
# ================================================================
# 注意：GPT框架会自动保证「第i个位置的输出只能看到前面1~i-1个位置的信息」，这就是它内部的 causal mask。
# 所以不需要再人为制作mask来控制因果方向。
class MaskedDecoder(nn.Module):
    def __init__(self, hidden_size, num_features, num_layers=2, nhead=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead,
            dim_feedforward=hidden_size * 4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruct = nn.Linear(hidden_size, num_features)

    def forward(self, hidden_states):
        ##hidden_states 是从 BertModel 输出的（即 out.last_hidden_state）,已经包含了位置编码信息,所以再输入解码器时 不需要重复加位置编码。
        out = self.decoder(hidden_states)
        return self.reconstruct(out)



class CNNUpsampler(nn.Module):
    """
    将 token-level hidden (B, T_token, d_model) 上采样到 point-level (B, sam_len, d_model)

    支持 chunk_size = 2^n，例如 8, 16, 32...
    通过动态堆叠 ConvTranspose1d，每层上采样 2 倍
    """

    def __init__(self, d_model=128, chunk_size=8):
        super().__init__()
        assert chunk_size & (chunk_size - 1) == 0, \
            f"chunk_size 必须是 2 的幂（如 8,16,32），当前 chunk_size={chunk_size}"

        n_layers = int(math.log2(chunk_size))  # 8 -> 3层, 16 -> 4层
        layers = []

        for _ in range(n_layers):
            layers.append(nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, h):
        """
        h: (B, T_token, d_model)
        return: (B, T_token*chunk_size, d_model)
        """
        h = h.transpose(1, 2)      # -> (B, d_model, T_token)
        h = self.net(h)            # -> (B, d_model, T_token*chunk_size)
        h = h.transpose(1, 2)      # -> (B, T_token*chunk_size, d_model)
        return h
    

class RepeatUpsampler(nn.Module):
    def __init__(self, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, h):  # (B,L,d)
        if self.chunk_size == 1:
            return h
        return h.repeat_interleave(self.chunk_size, dim=1)  # (B,T,d)


class PointRefiner(nn.Module):
    """
    只做点级细化，不改变长度。
    stack=层数可调，适配不同chunk_size：chunk越大，stack可越多
    深度可分离卷积：修边界、去噪，不会大范围抹平

    PointRefiner不能“上采样”，但非常适合做 上采样后的点级修复

    最推荐：repeat_interleave（确定性对齐） + PointRefiner（局部修复）
    原因：
    repeat_interleave 不会产生反卷积的周期性 artefact（测井里很常见）
    对薄层/边界更友好：一个 token 覆盖的 8 个点先完全对齐，再用 refiner 在点级做微调

    """
    def __init__(self, d_model, chunk_size=8, stack=None):
        super().__init__()
        if stack is None:
            if chunk_size <= 1:
                stack = 0   # 或 1，看你想要不要局部平滑
            else:
                # 经验：chunk=2/4 用1层，8/16 用2层，32 用3层
                stack = max(1, int(math.log2(chunk_size)) - 1)

        layers = []
        for _ in range(stack):
            layers += [
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B,T,d)
        if len(self.net) == 0:
            return x
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)
    



class LogGPT_Chunk16(nn.Module):
    """
    GPT2-based continuous-time model using chunk tokens (8 points -> 1 token)
    Model now expects input already chunked: (B, 64, 40)
    """

    def __init__(self, args, d_model=128, n_layer=2, n_head=2, dropout=0.1):
        super().__init__()

        # chunk 信息用于 upsample，但不作用于 forward 中的输入处理
        self.chunk_size = args.chunk_size
        self.num_features = len(args.fea)      # 5
        self.token_dim = self.num_features * self.chunk_size   # 40
        self.seq_len = args.sam_len // self.chunk_size         # 64

        n_classes = len(args.lith_code_map_name.keys())

        print(f"====== 使用的模型: {args.model_type}")
        print(f"====== 模型期望输入 shape: (B, {self.seq_len}, {self.token_dim})")
        print(f"====== 预训练的decoder网络: 一层nn.Linear({d_model, self.token_dim})")
        # print("====== 微调任务网络: MLP")
        print(f"====== 模型降低复杂度: n_layer={n_layer}, n_head={n_head}")  #exp30及之前的n_layer=4, n_head=4

        cfg = GPT2Config(
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=self.seq_len,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(cfg)

        # ⭐ 输入不再 chunk，直接输入 token_dim=40
        self.input_proj = nn.Sequential(
            nn.Linear(self.token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        ##self.decoder = MaskedDecoder(d_model, self.num_features)
        self.decoder = nn.Linear(d_model, self.token_dim)

        # ⭐ 微调阶段上采样到 512 点
        # self.upsample = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        # )
        #self.upsample = CNNUpsampler(d_model=d_model, chunk_size=self.chunk_size)

        self.upsample = RepeatUpsampler(self.chunk_size)
        self.refiner  = PointRefiner(d_model, chunk_size=self.chunk_size)



        # ⭐ 最终逐点分类
        # self.lstm = nn.LSTM(
        #     d_model, d_model,
        #     num_layers=2, batch_first=True,
        #     bidirectional=True
        # )

        self.lstm_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )



    def forward(self, x, model_task="pretrain", attention_mask=None):

        emb = self.input_proj(x)
        out = self.transformer(inputs_embeds=emb,
                            attention_mask=attention_mask,
                            return_dict=True)
        h = out.last_hidden_state  # (B, L, d_model)

        # ===== ① chunk-level 重构（预训练头，始终存在）=====
        recon = self.decoder(h)   # (B, L, token_dim)

        # ===== ② 分类分支 =====
        if model_task in ["finetune", "diretrain"]:

            if self.chunk_size > 1:
                # 上采样到 point-level
                h_up = self.upsample(h)    # (B,T,d)
                h_up = self.refiner(h_up)  # (B,T,d) 修边界
            else:
                h_up = h
                
            ##删除了lstm模块
            #h_lstm, _ = self.lstm(h_up)
            logits = self.lstm_classifier(h_up)
            return logits, recon

        elif model_task == "pretrain":
            return recon

        else:
            raise ValueError(f"====== ⚠️ Unknown model_task: {model_task}")








