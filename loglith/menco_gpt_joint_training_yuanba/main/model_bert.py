import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import math




def sample_bert_mask_weighted(mask_tokens, mask_ratio=0.3, ensure_one_mask=True, eps=1e-6):
 
    """
    在每个样本的有效 token 里，随机挑出一部分 token 做 BERT-style 的 mask（预测目标）

    mask_tokens: (B,L,D) 1=有效 0=缺失
    返回 bert_mask: (B,L) bool
    思想：token 被mask的概率 ∝ 有效维度比例

    return:
        bert_mask: (B,L) bool  True=被mask。  = True 表示这个 token 要被 BERT mask（也就是要模型去预测它）
        bert_mask: 表示 BERT 要预测的 token 位置（从 valid_tok 里采样）
    """
    B, L, D = mask_tokens.shape
    device = mask_tokens.device


    valid_frac = mask_tokens.float().mean(dim=-1)  # (B,L) in [0,1]   mean(dim=-1) 是对最后一维 D 求平均，也就是：每个 token 内有效维度占比
    # 至少有一个维度有效  True：这个 token 里至少有一个有效维度（不是全缺失） False：这个 token 全部维度都缺失（完全没信息）
    valid_tok = valid_frac > 0.0                  

    # 概率缩放到 [0,1]：有效越多越容易被选
    # 你可以用 valid_frac**p 让分布更尖锐，p=1~3
    #把 valid_frac 变成一个 0~1 的相对权重，让“有效维度更多的 token 更容易被 mask”。 
    #“在同一个样本内部，哪个 token 信息最完整，就给它最高的采样权重=1；信息越残缺，权重越接近0。”
    #valid_frac.max(dim=1, keepdim=True).values 在每个样本（每个 b）内部，找出它这一行 L 个 token 里 valid_frac 的最大值。
    #valid_frac / max(...) 做归一化：把本行最大的 valid_frac 变成 1，其他 token 就变成小于 1 的比例。
    prob = (valid_frac / (valid_frac.max(dim=1, keepdim=True).values + eps)).clamp(0,1)
    # 最终采样概率 = mask_ratio * prob
    rand = torch.rand((B, L), device=device)
    bert_mask = (rand < (mask_ratio * prob)) & valid_tok  #只有 同时满足：随机被选中 + token 有效 才能被 mask


    if ensure_one_mask: #为了兜底：每个样本至少 mask 1 个有效 token。
        has = bert_mask.any(dim=1)                 # (B,)
        need = (~has) & (valid_tok.any(dim=1))     # 有有效token但没mask到   need 的意义：找出需要“强制补一个 mask”的样本行。
        if need.any():
            #强制 mask生成

            score = torch.rand((B, L), device=device)
            score = score.masked_fill(~valid_tok, -1.0)
            idx = score.argmax(dim=1)
            bert_mask[need, idx[need]] = True

    return bert_mask




def sample_span_mask_vectorized(
    mask_tokens: torch.Tensor,       # (B, L, D) 1=有效 0=缺失
    chunk_size: int,
    mask_ratio: float = 0.3,         # 目标 mask 比例（在 valid token 上）
    mean_span_len: float = 24.0,      # span 平均长度
    max_span_len: int = 72,          # span 最大长度
    ensure_one_mask: bool = True,
    eps: float = 1e-6,
    
):
    """
    返回：
      bert_mask: (B, L) bool   True=该token被mask（需要预测）
      valid_tok: (B, L) bool   True=该token至少有一个有效维度
    说明：
      - 只在 valid_tok 上“生效”，最终会 bert_mask &= valid_tok
      - span 通过差分数组构造，无for循环
    """
    assert mask_tokens.dim() == 3
    B, L, D = mask_tokens.shape
    device = mask_tokens.device

    assert mean_span_len >= chunk_size, "mean_span_len 应该 >= chunk_size"
    assert max_span_len >= chunk_size, "max_span_len 应该 >= chunk_size"
    mean_span_len = int(mean_span_len/chunk_size)
    max_span_len = int(max_span_len/chunk_size)

    # token有效性：至少一个维度有效
    valid_tok = (mask_tokens.float().mean(dim=-1) > 0.0)  # (B,L) bool
    #统计每条序列里 valid token 数量：valid_cnt
    valid_cnt = valid_tok.sum(dim=1).clamp(min=1)         # (B,)

    # 每个样本希望mask的token数量（近似）
    target_mask_cnt = (valid_cnt.float() * mask_ratio).clamp(min=1.0)  # (B,)

    # 估计每个样本需要多少个span（标量 K），为了向量化固定K
    # K ≈ 目标mask数 / 平均span长度
    K = int(math.ceil((mask_ratio * L) / max(mean_span_len, 1.0))) + 1
    K = max(1, min(K, L))  # 防止太夸张

    # span长度：用几何分布的近似（更偏短span，更像地层连续遮挡）
    # geom(p) 的均值 = 1/p -> 令 p = 1/mean_span_len
    p = 1.0 / max(mean_span_len, 1.0)
    # 采样 U~(0,1)，geom_len = floor(log(1-U)/log(1-p)) + 1
    U = torch.rand((B, K), device=device).clamp(min=eps, max=1-eps)
    geom_len = (torch.log(1 - U) / torch.log(torch.tensor(1 - p, device=device))).floor().long() + 1
    span_len = geom_len.clamp(min=1, max=max_span_len)  # (B,K)

    # span起点：均匀采样 [0, L-1]
    start = torch.randint(low=0, high=L, size=(B, K), device=device)  # (B,K)
    end = (start + span_len).clamp(max=L)                              # (B,K), 右开区间

    # 差分数组 diff: (B, L+1)
    diff = torch.zeros((B, L + 1), device=device, dtype=torch.int32)

    # scatter_add：diff[b, start]+=1, diff[b, end]-=1 （无for循环）
    ones = torch.ones((B, K), device=device, dtype=torch.int32)
    diff.scatter_add_(dim=1, index=start, src=ones)
    diff.scatter_add_(dim=1, index=end,   src=-ones)

    covered = diff.cumsum(dim=1)[:, :L]  # (B,L) 覆盖次数
    bert_mask = covered > 0              # (B,L) bool

    # 只在有效token上生效
    bert_mask = bert_mask & valid_tok

    # （可选）把mask数量“压”到更接近目标：如果mask过多，用topk裁剪（仍无for循环）
    # 注意：topk裁剪会破坏“连续段”的纯粹性，但可以帮助比例更准。
    # 默认不开；你如果特别在意mask_ratio稳定，可以打开下面开关。
    # ---------
    # clip_to_target = False
    # if clip_to_target:
    #     # 用随机分数保留 target_mask_cnt 个mask位置
    #     score = torch.rand((B, L), device=device)
    #     score = score.masked_fill(~bert_mask, -1.0)
    #     k = target_mask_cnt.long().clamp(min=1, max=L)  # (B,)
    #     # 计算每行阈值：第k大
    #     kth = torch.topk(score, k=k.max().item(), dim=1).values  # (B,kmax)
    #     # 每行取第k个阈值（索引k-1）
    #     thresh = kth.gather(1, (k - 1).unsqueeze(1)).squeeze(1)  # (B,)
    #     bert_mask = (score >= thresh.unsqueeze(1)) & bert_mask
    # ---------

    # ensure_one_mask：每个样本至少mask 1个有效token（无for循环）
    if ensure_one_mask:
        has = bert_mask.any(dim=1)          # (B,)
        need = (~has) & valid_tok.any(dim=1)
        if need.any():
            score = torch.rand((B, L), device=device).masked_fill(~valid_tok, -1.0)
            idx = score.argmax(dim=1)       # (B,)
            bert_mask[need, idx[need]] = True

    return bert_mask, valid_tok


def apply_bert_corruption_on_emb(
    emb, x_tokens, bert_mask, input_proj, mask_token,
    prob_mask=0.8, prob_random=0.1 #
):
    """
    emb: (B,L,d) = input_proj(x_tokens)
    x_tokens: (B,L,D) 原始token（用于生成random token）
    bert_mask: (B,L) bool   True 的位置表示“这是要预测的 token”
    input_proj: token->emb 的投影层
    mask_token: (1,1,d) learnable  代表 [MASK]


    emb_corrupt[do_mask] = mask_token.expand(B, L, d)[do_mask]含义如下：
    mask_token 原本是 (1,1,d)，代表同一个 [MASK] 向量。
    mask_token.expand(B, L, d) 不复制数据地把它“广播成” (B, L, d)，方便和 emb_corrupt 对齐。
    emb_corrupt[do_mask]：当你用 (B,L) 的布尔 mask 索引 (B,L,d) 张量时，它会选出所有 True 对应的行，得到形状 (N_mask, d)。
    同理 mask_token.expand(...)[do_mask] 也是 (N_mask, d)。
    这句赋值：把所有需要 mask 的 token embedding 替换为同一个 mask_token。
    “把要预测的位置，80% 直接盖住变成 [MASK]。”


    返回 emb_corrupt: (B,L,d)
    """
    B, L, d = emb.shape
    device = emb.device
    emb_corrupt = emb.clone()

    r = torch.rand((B, L), device=device)

    # 80% -> [MASK]
    do_mask = (r < prob_mask) & bert_mask  #(B, L)  表示：哪些位置要被替换成 mask_token。
    if do_mask.any():
        emb_corrupt[do_mask] = mask_token.expand(B, L, d)[do_mask]

    # 10% -> random token
    do_rand = (r >= prob_mask) & (r < prob_mask + prob_random) & bert_mask  #do_rand表示哪些位置要被替换成随机 token embedding。
    if do_rand.any():
        #下述两句意味着：会从 整个 mini-batch 的任意样本、任意位置抽一个 token 作为“随机 token”。
        rand_b = torch.randint(0, B, (B, L), device=device) #rand_b 给每个位置 (b, l) 随机选一个 batch 索引（0..B-1）
        rand_l = torch.randint(0, L, (B, L), device=device) #rand_l 给每个位置随机选一个序列索引（0..L-1）
        
        #高级索引：对每个 (b,l)，取 x_tokens[rand_b[b,l], rand_l[b,l], :]
        #结果 rand_tok 形状是 (B, L, D)，每个位置都是随机抽来的 token 向量。
        rand_tok = x_tokens[rand_b, rand_l]       # (B,L,D)
        
        #把随机 token 向量投影成 embedding，得到 (B,L,d)
        rand_emb = input_proj(rand_tok)           # (B,L,d)

        #只在 do_rand=True 的位置，用随机 token embedding 替换原 embedding。
        emb_corrupt[do_rand] = rand_emb[do_rand]

    # 剩下 10% -> keep original emb (still predict)
    return emb_corrupt



def masked_mse_loss_token_weighted(pred, target, bert_mask, mask_tokens):
    
    """
    pred,target: (B,L,D)
    bert_mask: (B,L) bool  True=这些token参与loss
    mask_tokens: (B,L,D) 1=原始有效 0=原始缺失（可选）

    mask_tokens必须要传入参数，否则曲线原有缺失会参与计算loss。
    """

    diff = (pred - target) ** 2
    diff = diff * mask_tokens  # 过滤缺失维度

    # 每个token的有效维度数
    denom = mask_tokens.sum(dim=-1).clamp(min=1.0)     # (B,L) clamp(min=1) 是为了防止 denom=0（全缺失）导致除零。

    #把这个 token 的有效维度误差加起来，再除以有效维度数 → 得到“这个 token 每个有效维度的平均 MSE”。
    token_mse = diff.sum(dim=-1) / denom               # (B,L) 

    masked_token_mse = token_mse[bert_mask]
    if masked_token_mse.numel() == 0:
        return pred.sum() * 0.0
    return masked_token_mse.mean()







class CNNUpsampler(nn.Module):
    """
    将 token-level hidden (B, T_token, d_model) 上采样到 point-level (B, sam_len, d_model)

    支持 chunk_size = 2^n，例如 8, 16, 32...
    通过动态堆叠 ConvTranspose1d，每层上采样 2 倍

    ConvTranspose 容易产生 checkerboard / 周期性纹理（1D也会有）
    会“扩散”边界，使薄层变厚、夹层被抹掉
    在岩性不均衡任务里，小类往往就是薄层，最怕这种扩散

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




class LogBERT(nn.Module):
    """
    BERT-based continuous-time model using chunk tokens
    干净封装：
      - pretrain_step(): 内部完成 mask采样、corruption、forward、loss
      - forward_finetune(): 做分类（你也可以同时输出 recon）
    """

    def __init__(self, args, d_model=128, n_layer=3, n_head=2, dropout=0.1):
        super().__init__()

        self.chunk_size = args.chunk_size
        self.num_features = len(args.fea)
        self.token_dim = self.num_features * self.chunk_size
        self.seq_len = args.sam_len // self.chunk_size

        self.num_classes = len(args.lith_code_map_name.keys())

        print("====== 使用的模型: LogBERT")
        print(f"====== 输入 shape: (B, {self.seq_len}, {self.token_dim})")
        print(f"====== 模型降低复杂度: n_layer={n_layer}, n_head={n_head}")  #exp30及之前的n_layer=4, n_head=4

        print(f"====== 预训练任务: Masked Chunk_size={self.chunk_size} Reconstruction (BERT-style)")
        print("====== 当经过chunk化时，注意 PointRefiner 如何进行使用")

        # ✅ BERT Config
        cfg = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=self.seq_len,
            type_vocab_size=1
        )
        self.encoder = BertModel(cfg)

        # ✅ 输入投影 (token_dim -> d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(self.token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # ✅ BERT-style learnable [MASK]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # ✅ 预训练重构头 (d_model -> token_dim)
        self.recon_head = nn.Linear(d_model, self.token_dim)

        # # ✅ upsample head (chunk-level -> point-level)
        # self.upsample = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU()
        # )
        # self.upsample = CNNUpsampler(d_model=d_model, chunk_size=self.chunk_size)
        # ✅ 推荐：repeat + refiner
        self.upsample = RepeatUpsampler(self.chunk_size)
        self.refiner  = PointRefiner(d_model, chunk_size=self.chunk_size)

        # ✅ 分类头 (逐点分类)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.num_classes)
        )



    # -------- pretrain (封装版) --------
    def pretrain_step(self, x_tokens, mask_tokens, attention_mask=None,
                      ):
        """
        x_tokens:   (B,L,D)
        mask_tokens:(B,L,D) 1=原始有效 0=缺失
        attention_mask:(B,L) 1=有效token 0=padding（可选）
        返回：loss, recon, bert_mask
        """
        # # 1) 仅在“足够有效”的 token 上采样 BERT mask
        ##bert_mask = sample_bert_mask_weighted(mask_tokens)
        bert_mask, valid_tok = sample_span_mask_vectorized(mask_tokens, 
                                                           chunk_size=self.chunk_size)
        # 2) emb
        emb = self.input_proj(x_tokens)  # (B,L,d)

        # 3) corruption on emb
        emb_corrupt = apply_bert_corruption_on_emb(
                        emb, x_tokens, bert_mask, self.input_proj, self.mask_token,
                    )
        if attention_mask is None:  
            attention_mask = valid_tok.long()  #全缺失 token 在 self-attention 中被当作 padding，不参与注意力计算。例如：某深度点上所有曲线都缺失
        # 4) encode
        out = self.encoder(inputs_embeds=emb_corrupt,
                           attention_mask=attention_mask,
                           return_dict=True)
        h = out.last_hidden_state  # (B,L,d)

        # 5) recon
        recon = self.recon_head(h)  # (B,L,D)  #应该重构 点级 而不是 chunk级的    

        # 6) loss：只在 bert_mask token 上，并剔除原始缺失维度
        loss = masked_mse_loss_token_weighted(recon, x_tokens, bert_mask, mask_tokens)

        # 一些诊断信息（可写tensorboard）
        info = {
            "mask_ratio_actual": bert_mask.float().mean().item(),
        }

        return loss, recon, bert_mask, info


    # -------- finetune --------
    def finetune_step(self, x_tokens, mask_tokens, attention_mask=None):
        valid_tok = (mask_tokens.float().mean(dim=-1) > 0.0)  # (B,L) bool
        
        emb = self.input_proj(x_tokens)

        if attention_mask is None:
            attention_mask = valid_tok.long()
        out = self.encoder(inputs_embeds=emb, attention_mask=attention_mask, return_dict=True)
        
        h = out.last_hidden_state

        h_up = self.upsample(h)    # (B,T,d)
        h_up = self.refiner(h_up)  # (B,T,d) 修边界
        logits = self.classifier(h_up)
        return logits




    # def forward(self, x_tokens, attention_mask=None, model_task="pretrain"):
    #     """
    #     x_tokens: (B, L, token_dim)
    #     attention_mask: (B, L)  1=valid, 0=pad
    #     """

    #     emb = self.input_proj(x_tokens)  # (B, L, d_model)

    #     out = self.encoder(inputs_embeds=emb,
    #                        attention_mask=attention_mask,
    #                        return_dict=True)
    #     #print("====== encoder 输出 是last_hidden_state这种吗? 有return_dict=True架构吗")
    #     h = out.last_hidden_state  # (B, L, d_model)

    #     # ✅ reconstruction always computed
    #     recon = self.recon_head(h)  # (B, L, token_dim)

    #     # ===== ② 分类分支 =====
    #     if model_task in ["finetune", "diretrain"]:

    #         if self.chunk_size > 1:
    #             # 上采样到 point-level
    #             h_up = self.upsample(h)
    #         else:
    #             h_up = h
                
    #         ##删除了lstm模块
    #         #h_lstm, _ = self.lstm(h_up)
    #         logits = self.classifier(h_up)
    #         return logits, recon

    #     elif model_task == "pretrain":
    #         return recon

    #     else:
    #         raise ValueError(f"====== ⚠️ Unknown model_task: {model_task}")

