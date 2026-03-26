import torch
import os
import json
import random 
import numpy as np
import torch.nn as nn

from sklearn.metrics import f1_score  
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import time
import shutil
import torch.nn.functional as F

print("======= 在预训练和微调的for循环中更新了 每epoch的种子!!!")


# -----------------------
# Loss helpers
# -----------------------
def masked_mse_loss(pred, target, mask):
    """
    pred, target, mask: [B, T, C]
    mask: 1有效, 0无效
    """
    loss_mat = (pred - target) ** 2
    loss = (loss_mat * mask).sum() / (mask.sum() + 1e-8)
    return loss




def chunk_for_pretrain(xb_ori, masko, args):
    """
    GPT-style chunk autoregressive pretraining

    xb_ori: (B, T, C)
    masko:  (B, T, C)
    return:
        x_in:   (B, L-1, token_dim)
        y_out:  (B, L-1, token_dim)
        mask:   (B, L-1, token_dim)
    """
    B, T, C = xb_ori.shape
    chunk_size = args.chunk_size
    token_dim = C * chunk_size
    seq_len = T // chunk_size

    T_use = seq_len * chunk_size
    xb_ori = xb_ori[:, :T_use]
    masko  = masko[:, :T_use]

    # (B, L, chunk, C)
    xb_chunks = xb_ori.view(B, seq_len, chunk_size, C)
    mask_chunks = masko.view(B, seq_len, chunk_size, C)

    # flatten
    x_tokens = xb_chunks.reshape(B, seq_len, token_dim)
    mask_tokens = mask_chunks.reshape(B, seq_len, token_dim)

    if args.stage == "pretrain":  ##注意预训练需要偏移，微调岩性分类时不需要
#         ##print("======. args.stage", args.stage)
        # GPT shift
        x_in  = x_tokens[:, :-1, :]
        y_out = x_tokens[:, 1:, :]
        mask  = mask_tokens[:, 1:, :]
    else:
        x_in  = x_tokens
        y_out = x_tokens
        mask  = mask_tokens

    return x_in, y_out, mask


def chunkify_tokens(xb_ori, masko, args):
    """
    把原始连续曲线切 chunk 并 flatten 成 token。

    xb_ori: (B, T, C)
    masko:  (B, T, C)  1=有效，0=缺失

    return:
      x_tokens:   (B, L, token_dim)
      mask_tokens:(B, L, token_dim)
    """
    B, T, C = xb_ori.shape
    chunk_size = args.chunk_size
    token_dim = C * chunk_size
    seq_len = T // chunk_size
    T_use = seq_len * chunk_size

    xb_ori = xb_ori[:, :T_use]
    masko  = masko[:, :T_use]

    # (B, L, chunk, C)
    xb_chunks   = xb_ori.view(B, seq_len, chunk_size, C)
    mask_chunks = masko.view(B, seq_len, chunk_size, C)

    # flatten -> (B, L, token_dim)
    x_tokens    = xb_chunks.reshape(B, seq_len, token_dim)
    mask_tokens = mask_chunks.reshape(B, seq_len, token_dim)

    return x_tokens, mask_tokens


def build_ar_next_target(x_tokens, mask_tokens):
    """
    AR next-token 目标：
      pred at t  -> target at t+1
    所以用于loss的对齐是：
      pred = recon[:, :-1]
      tgt  = x_tokens[:, 1:]
      mask = mask_tokens[:, 1:]
    """
    tgt  = x_tokens[:, 1:, :]
    mask = mask_tokens[:, 1:, :]
    return tgt, mask

def pretrain_network(args, model, train_loader, val_loader, device,
                     epoch_inte=20, history_save_inte=500):
    """
    自监督预训练阶段：
    ⭐ 使用 chunk token（8 点 → 1 token）进行曲线重建预训练
    - 模型输入: (B, seq_len=64, token_dim=40)
    - 模型输出: (B, seq_len=64, C=5)
    - 标签: 每个 chunk 内按 mask 加权平均后的 5 维曲线
    """

    experiment_name = args.experiment_name   
    print(f"====== 当前预训练实验：{experiment_name} ======")

    # --- tensorboard ---
    tb_dir = f"../tensorboard/{experiment_name}/"
    clear_tensorboard_logs(log_dir=tb_dir)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # --- model save directory ---
    model_dir = f"../model/{experiment_name}/"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_pretrain_model.pth")

    # --- figure directory ---
    fig_dir = f"../fig/loss/{experiment_name}/"
    os.makedirs(fig_dir, exist_ok=True)

    # --- history directory ---
    history_dir = f"../history/{experiment_name}/"
    os.makedirs(history_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": []}

    stime = time.time()
    for ep in range(1, args.epochs + 1):
        np.random.seed(42 + ep)
        torch.manual_seed(42 + ep)
        torch.cuda.manual_seed_all(42 + ep)

        # ---------------------- Training ----------------------
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            xb_ori, masko = batch["fea_log"], batch["masko"]  # (B, T=512, C)

            # chunk 化
            x_chunk, tgt_chunk, mask_chunk = chunk_for_pretrain(
                xb_ori, masko, args
            )  # x_chunk:(B,64,40), tgt_chunk/mask_chunk:(B,64,5)

            optimizer.zero_grad()
            # 预训练任务：重建 chunk-level 曲线
            preds = model(x_chunk, model_task=args.stage)  # (B,64,5)

            loss = masked_mse_loss(preds, tgt_chunk, mask_chunk)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # ---------------------- Validation ----------------------
        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                xb_ori, masko = batch["fea_log"], batch["masko"]

                x_chunk, tgt_chunk, mask_chunk = chunk_for_pretrain(
                    xb_ori, masko, args
                )

                preds = model(x_chunk, model_task=args.stage)
                loss = masked_mse_loss(preds, tgt_chunk, mask_chunk)
                val_total += loss.item()

        val_loss = val_total / len(val_loader)
        history["val_loss"].append(val_loss)

        # ---------------------- Logging ----------------------
        etime = time.time()
        print(f"[Pretrain {experiment_name}] Epoch {ep}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {etime-stime:.2f}s")

        writer.add_scalar("Loss/Train", train_loss, ep)
        writer.add_scalar("Loss/Val", val_loss, ep)

        # ---------------------- Save Best Model ----------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            best_epoch = ep
            print(f"--> Saved Best Pretrain Model at epoch {ep} (val_loss={val_loss:.4f})")

        # 周期性保存
        if ep % epoch_inte == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"pretrain_epoch_{ep}.pth"))

        # ---------------------- Periodic History Save ----------------------
        if ep % history_save_inte == 0:
            history_tmp_path = os.path.join(history_dir, "pretrain_tmp_history.json")
            with open(history_tmp_path, "w") as f:
                json.dump({
                    "last_epoch": ep,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "history": history
                }, f, indent=4)
            print(f"[Checkpoint] TMP History saved at epoch {ep} -> {history_tmp_path}")

    writer.close()

    # =====================================================
    # 4. 画 Loss 曲线（与 finetune 统一结构）
    # =====================================================
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Pretrain Loss Curve ({experiment_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "pretrain_loss.jpg")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"====== 预训练 Loss 曲线已保存至: {fig_path}")

    # =====================================================
    # 5. 保存 history JSON
    # =====================================================
    history_path = os.path.join(history_dir, "pretrain_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "total_epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": history
        }, f, indent=4)
    print(f"====== 预训练 history 已保存到: {history_path}")

    print("======== Pretraining finished!!!")
    print(f"======== 最好模型 epoch = {best_epoch} (val_loss={best_val_loss:.4f}) ========")



def train_network(args, model, train_loader, val_loader, num_class, 
                  device=None, epoch_inte=20, freeze_encoder=True,
                  history_save_inte=500,
                  lambda_pre=0.1):
    
    experiment_name = args.experiment_name
    print(f"====== 当前实验：{experiment_name} ======")
    print("====== torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 已启用，进行梯度裁剪")

    tensorboard_dir = f"../tensorboard/{experiment_name}/"
    ##清除旧的tensorboard
    clear_tensorboard_logs(log_dir = tensorboard_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    ##history path
    history_dir = f"../history/{experiment_name}/"
    os.makedirs(history_dir, exist_ok=True)

    # ====== 根据 loss_type 选择损失函数 ======
    if args.loss_type == "ce":
        print("====== 使用 CrossEntropyLoss()")
        criterion = nn.CrossEntropyLoss()

    elif args.loss_type == "focal":
        print("====== 使用 FocalLoss(gamma=2.0)")
        criterion = FocalLoss(gamma=2.0, reduction="mean")   #alpha=class_weights, 

    elif args.loss_type == "balanced_focal":
        print("====== 使用 带类别权重的FocalLoss()")
        class_counts = torch.tensor(args.lith_weight, dtype=torch.float)
        alpha = 1.0 / torch.sqrt(class_counts)   # 稀有类更大的权重
        alpha = alpha / alpha.sum()
        print(f"类别权重 alpha = {alpha.tolist()}")
        criterion = FocalLoss(gamma=2.0, alpha=alpha, reduction="mean")

    elif args.loss_type == "focal_tversky":
        print("====== 使用 FocalTverskyLoss()")
        lith_weight = torch.tensor(args.lith_weight, dtype=torch.float)
        print(f"====== 类别权重 lith_weight = {lith_weight}")
        criterion = FocalTverskyLossBLC(
                                        alpha=0.3, beta=0.7, gamma=1.3,  #应该设置1.25～2才对。
                                        eps=1e-6,
                                        class_weights=lith_weight,     # 可选: (C,)
                                        ignore_index=None       # 可选: 忽略某些标签
                                        )
        print("====== 注意查看gamma权重是否使用1.25～2.0的范围，这个范围通常对不平衡分类更有效，class_weight是否正确加载了lith_weight")

    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    
    ####微调. 不冻结，但弱化重构学习率
    encoder_params = []
    head_params = []
    print("====== ⚠️ 微调：不冻结, 一定注意模型架构变化，这里微调的架构可能需要随之变化")
    for name, p in model.named_parameters():
        if "transformer" in name or "input_proj" in name:
            encoder_params.append(p)
        else:
            head_params.append(p)
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.main_net_lr},
        {"params": head_params, "lr": args.lr}
    ])
    if args.stage != "diretrain":
        # ====== ✅ 这里：创建完 optimizer 立刻 resume ======
        if hasattr(args, "resume_ckpt") and args.resume_ckpt is not None and os.path.exists(args.resume_ckpt):
            print(f"====== ✅ Resuming from checkpoint: {args.resume_ckpt}")
            resume_epoch = load_checkpoint_for_resume(args.resume_ckpt, model, optimizer, device=device)
            print(f"====== ✅ resume_epoch={resume_epoch} (仅用于记录，不影响你的for循环)")

            # 把 optimizer.state 的 tensor 搬到 GPU
            ##当加载的是预训练模型，因为是纯state_dict，没有optimizer的state，所以这里不会执行
            ##当加载的是微调断点时，不是空，所以会执行
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            print("====== args.resume_ckpt", args.resume_ckpt)
            print("====== ⚠️⚠️⚠️ No pretrained/resume model found, training from scratch!")
            

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"======= 可训练参数: {trainable_params:,} / 总参数: {total_params:,} "
        f"({trainable_params/total_params:.2%} 可训练)")

    print("\n====== 参数分组与学习率 ======")
    for i, group in enumerate(optimizer.param_groups):
        lr = group["lr"]
        print(f"\n--- Param Group {i} | lr = {lr} ---")
        for p in group["params"]:
            for name, param in model.named_parameters():
                if param is p:
                    print(f"{name:60s} | requires_grad={param.requires_grad}")
                    break


    # 在 train_network 开头初始化
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    model_dir = f"../model/{experiment_name}/"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pth")


    # 记录训练过程
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_cls": [],   # ⭐新增
        "train_loss_pre": [],   # ⭐新增
        "val_loss_cls": [],     # ⭐新增
        "val_loss_pre": [],     # ⭐新增
        "train_acc": [],
        "val_acc": [],
        "train_per_class_acc": [],
        "val_per_class_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_per_class_f1": [],
        "val_per_class_f1": [],
        "lambda_pre": []    # ⭐新增
    }
    stime = time.time()
    for epoch in range(1, args.epochs + 1):

        # 每个 epoch 改一个确定性的偏移种子  使得不同epoch之间的样本不同
        np.random.seed(42 + epoch)
        torch.manual_seed(42 + epoch)
        torch.cuda.manual_seed_all(42 + epoch)

        # atime = time.time()
        train_loss, train_loss_cls, train_loss_pre, train_acc, train_per_class_acc, train_f1, train_per_class_f1 = train_one_epoch(
            args, model, train_loader, criterion, optimizer, device, num_class, lambda_pre=lambda_pre,
        )

        # train_time = time.time()-atime
        # print(f"训练消耗的时间为:{train_time}")

        val_loss, val_loss_cls, val_loss_pre, val_acc, val_per_class_acc, val_f1, val_per_class_f1 = validate_one_epoch(
            args, model, val_loader, criterion, device, num_class, lambda_pre=lambda_pre,
        )


        r = 0.05  # 目标占比 5%
        lambda_pre = torch.clamp(
                    torch.tensor(r * train_loss_cls / (train_loss_pre + 1e-8)),
                    1.0, 20.0
                ).to(device)
        
        # val_time = time.time()-atime-train_time
        # print(f"验证消耗的时间为:{val_time}")

        # 保存历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_loss_cls"].append(train_loss_cls)   # ⭐新增
        history["train_loss_pre"].append(train_loss_pre)   # ⭐新增
        history["val_loss_cls"].append(val_loss_cls)       # ⭐新增
        history["val_loss_pre"].append(val_loss_pre)       # ⭐新增
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_per_class_acc"].append(train_per_class_acc)
        history["val_per_class_acc"].append(val_per_class_acc)

        # ===== 新增 F1 =====
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_per_class_f1"].append(train_per_class_f1)
        history["val_per_class_f1"].append(val_per_class_f1)
        history["lambda_pre"].append(lambda_pre.item())    # ⭐新增


        # TensorBoard 可视化
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Loss/Train_cls", train_loss_cls, epoch)   # ⭐新增
        writer.add_scalar("Loss/Train_pre", train_loss_pre, epoch)   # ⭐新增
        writer.add_scalar("Loss/Val_cls", val_loss_cls, epoch)       # ⭐新增
        writer.add_scalar("Loss/Val_pre", val_loss_pre, epoch)       # ⭐新增
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)   
        writer.add_scalar("F1/Val", val_f1, epoch)
        writer.add_scalar("Lamda_pre", lambda_pre, epoch)    # ⭐新增

        for c in range(num_class):
            writer.add_scalar(f"ClassAcc/Train_class_{c}", train_per_class_acc[c], epoch)
            writer.add_scalar(f"ClassAcc/Val_class_{c}", val_per_class_acc[c], epoch)

        etime = time.time()

        # 打印信息
        #if epoch % epoch_inte == 0 or epoch == 1:
        print(f"[{experiment_name} Epoch {epoch}/{args.epochs}] | "
                f"Train: total={train_loss:.4f} cls={train_loss_cls:.4f} pre={train_loss_pre:.4f} acc={train_acc:.4f} || "
                f"Val: total={val_loss:.4f} cls={val_loss_cls:.4f} pre={val_loss_pre:.4f} acc={val_acc:.4f} || "
                f"Lamda_pre(Epoch+1): {lambda_pre:.4f} | "
                f"Time: {etime-stime:.3f}")

        # 保存周期性模型
        if epoch % epoch_inte == 0:
            # torch.save(model.state_dict(), os.path.join(model_dir, f"epoch_{epoch}.pth"))
            ckpt_path = os.path.join(model_dir, f"epoch_{epoch}.pth")
            save_checkpoint({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # 如果你有 scheduler，也可以加：
                # "scheduler_state_dict": scheduler.state_dict(),
                # 如果你用 AMP，也可以加：
                # "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,   # 记录一下不影响你 loop，从1开始也没事
            }, ckpt_path)


        if (val_loss < best_val_loss) or (np.isclose(val_loss, best_val_loss) and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc
            }, best_model_path)
            best_epoch = epoch  #标记一下最好的epoch
            print(f"--> New best model saved (epoch {epoch}) val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")  #[{experiment_name}] 

        # ---------------------- Periodic History Save ----------------------
        if epoch % history_save_inte == 0:
            history_tmp_path = os.path.join(history_dir, "training_tmp_history.json")
            history = convert_to_serializable(history)
            with open(history_tmp_path, "w") as f:
                json.dump({
                    "last_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "history": history
                }, f, indent=4)
            print(f"[Checkpoint] TMP History saved at epoch {epoch} -> {history_tmp_path}")



    writer.close()
    print(f"====== 训练完成，最优模型已保存到:{best_model_path}", "\n",
          f"====== 最好模型的epoch是:{best_epoch}")
    
    # 绘制并保存图像
    fig_dir = f"../fig/loss/{experiment_name}/"
    plot_loss_acc_curves(history, num_class, save_dir=fig_dir)

    # ===== 保存 history 到 JSON =====
    history_path = os.path.join(history_dir, "training_history.json")
    history = convert_to_serializable(history)
    with open(history_path, "w") as f:
        json.dump({
            "total_epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": history
        }, f, indent=4)
    print(f"====== 训练过程 history data 已保存到: {history_path}")










def train_one_epoch(args, model, train_loader, criterion, optimizer, device, num_class,
                    lambda_pre=None):
    model.train()

    total_loss_sum = 0.0
    cls_loss_sum   = 0.0
    pre_loss_sum   = 0.0

    train_correct, train_total = 0, 0
    all_preds, all_targets = [], []

    for data in train_loader:
        data = {key: tensor.to(device) for key, tensor in data.items()}
        batch_x, target, masko = data['fea_log'], data['lith'], data['masko']   # (B,512,5), (B,512)

        # ===== chunkify =====
        x_tokens, mask_tokens = chunkify_tokens(batch_x, masko, args)          # (B,64,40), (B,64,40)

        optimizer.zero_grad()

        # ===== forward: outputs分类 + recon重构 =====
        outputs, recon = model(x_tokens, model_task=args.stage)                # outputs:(B,512,num_class), recon:(B,64,40)

        # ✅【具体位置：就在这里】forward 后立刻检查，不正常就跳过
        if (not torch.isfinite(outputs).all()) or (not torch.isfinite(recon).all()):
            print("====== outputs/recon nan/inf -> skip this batch")
            # 可选：保存坏 batch 方便复现
            # torch.save({"x_tokens": x_tokens.detach().cpu(),
            #             "target": target.detach().cpu(),
            #             "masko": masko.detach().cpu()}, "nan_forward_batch.pt")
            optimizer.zero_grad(set_to_none=True)
            continue


        # if not check_finite("x_tokens", x_tokens): 
        #     raise RuntimeError("train x_tokens not finite")
        # if not check_finite("mask_tokens", mask_tokens): 
        #     raise RuntimeError("train mask_tokens not finite")
        # if not check_finite("outputs", outputs):
        #     print("====== train x_tokens max/min:", x_tokens.abs().max().item(), x_tokens.min().item(), x_tokens.max().item())
        #     print("====== train batch_x max/min:", batch_x.abs().max().item(), batch_x.min().item(), batch_x.max().item())
            
        #     # 在 outputs 变 NaN 时调用
        #     check_params(model)
            
        #     torch.save({"train x_tokens": x_tokens.detach().cpu(),
        #                 "train targets": target.detach().cpu(),
        #                 "train masko": masko.detach().cpu()}, "nan_batch_debug.txt")
        #     raise RuntimeError("outputs not finite")
        # if not check_finite("recon", recon):
        #     torch.save({"train x_tokens": x_tokens.detach().cpu(),
        #                 "train targets": target.detach().cpu(),
        #                 "train masko": masko.detach().cpu()}, "nan_batch_debug.txt")
        #     raise RuntimeError("train recon not finite")


        # ===== 分类 loss =====
        loss_cls = criterion(outputs.view(-1, num_class), target.view(-1))

        # ===== AR next-token 重构 loss =====
        tgt_next, mask_next = build_ar_next_target(x_tokens, mask_tokens)      # (B,63,40)
        pred_next = recon[:, :-1, :]                                           # (B,63,40)
        loss_pre = masked_mse_loss(pred_next, tgt_next, mask_next)

        # ===== 总 loss =====
        #loss = loss_cls + lambda_pre * loss_pre
        loss = loss_cls #+ lambda_pre * loss_pre

        # ✅ 再加一层：loss 本身 NaN 也跳过
        if not torch.isfinite(loss):
            print("====== loss nan/inf -> skip this batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            print("====== grad_norm nan/inf -> skip this batch (no optimizer.step)")
            optimizer.zero_grad(set_to_none=True)
            continue   # ✅关键：跳过 optimizer.step()

        optimizer.step()

        # ===== accumulate =====
        total_loss_sum += loss.item()
        cls_loss_sum   += loss_cls.item()
        pre_loss_sum   += loss_pre.item()

        preds = outputs.argmax(dim=-1)  # (B,512)
        train_correct += (preds == target).sum().item()
        train_total += target.numel()

        all_preds.append(preds.detach())
        all_targets.append(target.detach())


        # if not torch.isfinite(loss_cls):
        #     print("loss_cls nan/inf")
        #     print(loss_cls, loss_pre, lambda_pre, torch.isfinite(outputs).all(), torch.isfinite(recon).all())
        #     break
        # if not torch.isfinite(loss_pre):
        #     print("loss_pre nan/inf")
        #     print(loss_cls, loss_pre, lambda_pre, torch.isfinite(outputs).all(), torch.isfinite(recon).all())
        #     break
        # if not torch.isfinite(outputs).all():
        #     print("outputs has nan/inf")
        #     print(loss_cls, loss_pre, lambda_pre, torch.isfinite(outputs).all(), torch.isfinite(recon).all())
        #     break
        # if not torch.isfinite(recon).all():
        #     print("recon has nan/inf")
        #     print(loss_cls, loss_pre, lambda_pre, torch.isfinite(outputs).all(), torch.isfinite(recon).all())
        #     break


    # ===== epoch stats =====
    avg_total_loss = total_loss_sum / len(train_loader)
    avg_cls_loss   = cls_loss_sum   / len(train_loader)
    avg_pre_loss   = pre_loss_sum   / len(train_loader)

    acc = train_correct / train_total
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    per_class_acc = compute_per_class_accuracy(all_preds, all_targets, num_class)

    f1_macro = f1_score(all_targets.view(-1).cpu().numpy(),
                        all_preds.view(-1).cpu().numpy(),
                        average='macro',  #weighted：大类占比高 → 权重大 → 对最终分数影响大  macro：（平均每一类的表现，不看样本数）
                        zero_division=0)
    per_class_f1 = f1_score(all_targets.view(-1).cpu().numpy(),
                            all_preds.view(-1).cpu().numpy(),
                            average=None,
                            labels=list(range(num_class)),
                            zero_division=0)

    # ⭐ 返回时把分类loss和重构loss也返回
    return avg_total_loss, avg_cls_loss, avg_pre_loss, acc, per_class_acc, f1_macro, per_class_f1



def validate_one_epoch(args, model, val_loader, criterion, device, num_class,
                       lambda_pre=None):
    model.eval()

    total_loss_sum = 0.0
    cls_loss_sum   = 0.0
    pre_loss_sum   = 0.0

    val_correct, val_total = 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data in val_loader:
            data = {key: tensor.to(device) for key, tensor in data.items()}
            batch_x, target, masko = data['fea_log'], data['lith'], data['masko']

            x_tokens, mask_tokens = chunkify_tokens(batch_x, masko, args)

            outputs, recon = model(x_tokens, model_task=args.stage)

            # if not check_finite("x_tokens", x_tokens): 
            #     raise RuntimeError("x_tokens not finite")
            # if not check_finite("mask_tokens", mask_tokens): 
            #     raise RuntimeError("mask_tokens not finite")
            # if not check_finite("outputs", outputs):
                
            #     print("x_tokens max/min:", x_tokens.abs().max().item(), x_tokens.min().item(), x_tokens.max().item())
            #     print("batch_x max/min:", batch_x.abs().max().item(), batch_x.min().item(), batch_x.max().item())
                
            #     # 在 outputs 变 NaN 时调用
            #     check_params(model)
                
            #     torch.save({"x_tokens": x_tokens.detach().cpu(),
            #                 "targets": target.detach().cpu(),
            #                 "masko": masko.detach().cpu()}, "nan_batch_debug.txt")
            #     raise RuntimeError("outputs not finite")
            # if not check_finite("recon", recon):
            #     torch.save({"x_tokens": x_tokens.detach().cpu(),
            #                 "targets": target.detach().cpu(),
            #                 "masko": masko.detach().cpu()}, "nan_batch_debug.txt")
            #     raise RuntimeError("recon not finite")

            loss_cls = criterion(outputs.view(-1, num_class), target.view(-1))

            tgt_next, mask_next = build_ar_next_target(x_tokens, mask_tokens)
            pred_next = recon[:, :-1, :]
            loss_pre = masked_mse_loss(pred_next, tgt_next, mask_next)

            # loss = loss_cls + lambda_pre * loss_pre
            loss = loss_cls #+ lambda_pre * loss_pre

            total_loss_sum += loss.item()
            cls_loss_sum   += loss_cls.item()
            pre_loss_sum   += loss_pre.item()

            preds = outputs.argmax(dim=-1)
            val_correct += (preds == target).sum().item()
            val_total += target.numel()

            all_preds.append(preds)
            all_targets.append(target)

            # if not torch.isfinite(loss_cls):
            #     print("验证集 loss_cls nan/inf")
            #     print(loss_cls, loss_pre, lambda_pre, torch.isfinite(outputs).all(), torch.isfinite(recon).all())
            #     break
            # if not torch.isfinite(loss_pre):
            #     print("loss_pre nan/inf")
            #     print(loss_cls, loss_pre, lambda_pre, torch.isfinite(outputs).all(), torch.isfinite(recon).all())
            #     break
            # if not torch.isfinite(outputs).all():
            #     print("验证集 outputs has nan/inf")
            #     print(loss_cls, loss_pre, lambda_pre, torch.isfinite(outputs).all(), torch.isfinite(recon).all())
            #     break
            # if not torch.isfinite(recon).all():
            #     print("验证集 recon has nan/inf")
            #     print(loss_cls, loss_pre, lambda_pre, torch.isfinite(outputs).all(), torch.isfinite(recon).all())
            #     break

    avg_total_loss = total_loss_sum / len(val_loader)
    avg_cls_loss   = cls_loss_sum   / len(val_loader)
    avg_pre_loss   = pre_loss_sum   / len(val_loader)

    acc = val_correct / val_total
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    per_class_acc = compute_per_class_accuracy(all_preds, all_targets, num_class)

    f1_macro = f1_score(all_targets.view(-1).cpu().numpy(),
                        all_preds.view(-1).cpu().numpy(),
                        average='macro',
                        zero_division=0)
    per_class_f1 = f1_score(all_targets.view(-1).cpu().numpy(),
                            all_preds.view(-1).cpu().numpy(),
                            average=None,
                            labels=list(range(num_class)),
                            zero_division=0)

    return avg_total_loss, avg_cls_loss, avg_pre_loss, acc, per_class_acc, f1_macro, per_class_f1


def check_finite(name, x):
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        print(f"x.shape: {x.shape}")
        print(f"[NaN/Inf] {name}: count={bad.numel()} example={bad.flatten()[:5]}")
        return False
    return True


def check_params(model):
    for n, p in model.named_parameters():
        print("====== p.min().item(), p.max().item():", p.min().item(), p.max().item())
        if p is None:
            continue
        if not torch.isfinite(p).all():
            bad = p[~torch.isfinite(p)]
            print(f"[NaN/Inf PARAM] {n} count={bad.numel()} example={bad.flatten()[:5]}")
            return False
    return True



def load_checkpoint_for_resume(ckpt_path, model, optimizer=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    # print(f"====== ckpt {ckpt}")

    # ✅ finetune checkpoint: dict with "model_state_dict"
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print("====== [resume ckpt] strict=True 加载 model_state_dict")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            print("====== [resume ckpt] 加载 optimizer_state_dict")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("====== [resume ckpt] 未找到 optimizer_state_dict，跳过优化器加载")

        return ckpt.get("epoch", None)

    # ✅ pretrain weights: pure state_dict
    print("====== [pretrain weights] strict=False 加载纯 state_dict")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("====== Missing:", missing)
    print("====== Unexpected:", unexpected)
    return None

def compute_per_class_accuracy(preds, targets, num_class):
    """计算每类的准确率，注意其实每类的准确率就是召回率"""
    correct_per_class = np.zeros(num_class, dtype=np.int64)
    total_per_class = np.zeros(num_class, dtype=np.int64)

    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    #print("====== 计算各类准确率时 preds.shape", preds.shape)

    for c in range(num_class):
        mask = (targets == c)
        correct_per_class[c] += np.sum((preds[mask] == c))
        total_per_class[c] += np.sum(mask)

    acc_per_class = correct_per_class / np.maximum(total_per_class, 1)  # 防止除0
    return acc_per_class



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: 聚焦因子
        alpha: tensor of shape (num_class,) 对每类加权，可为 None
        reduction: 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_class)
        targets: (batch_size,) long tensor
        """
        logpt = F.log_softmax(inputs, dim=1)  # log(p)
        pt = torch.exp(logpt)  # p

        # 选出真实类别的 pt 和 logpt
        targets = targets.view(-1, 1)
        logpt = logpt.gather(1, targets).view(-1)
        pt = pt.gather(1, targets).view(-1)

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.view(-1))
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class FocalTverskyLossBLC(nn.Module):
    """
    Focal Tversky Loss for multi-class classification (flattened sequence)
    logits:  (N, C)   where N = B*L
    targets: (N,)     int labels
    """
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        gamma=0.75,
        eps=1e-6,
        class_weights=None,   # Tensor shape (C,)
        ignore_index=None
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.ignore_index = ignore_index

        # class_weights: 作为 buffer 存储（自动跟随 device），但不参与训练
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32) \
                if not torch.is_tensor(class_weights) else class_weights.float()
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        logits:  (N, C)
        targets: (N,)
        """
        N, C = logits.shape
        probs = torch.softmax(logits, dim=1)  # (N, C)

        #print("====== logits.dtype, targets.dtype:", logits.dtype, targets.dtype)

        # 1) ignore_index mask
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index)      # (N,)
            probs = probs[mask]                        # (N_valid, C)
            targets = targets[mask]                    # (N_valid,)
            if targets.numel() == 0:
                # 全部被 mask 掉时避免 nan
                return torch.zeros([], device=logits.device, requires_grad=True)

        # 2) one-hot
        targets_onehot = F.one_hot(targets, num_classes=C).float()  # (N_valid, C)

        #print("====== targets_onehot.dtype:", targets_onehot.dtype)

        # 3) TP / FP / FN per class
        tp = (probs * targets_onehot).sum(dim=0)                   # (C,)
        fp = (probs * (1 - targets_onehot)).sum(dim=0)             # (C,)
        fn = ((1 - probs) * targets_onehot).sum(dim=0)             # (C,)

        # 4) Tversky index per class
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)  # (C,)
        loss_per_class = (1 - tversky).pow(self.gamma)  # (C,)

        # 5) class weighted mean
        if self.class_weights is not None:
            w = self.class_weights.to(loss_per_class.device)
            loss = (loss_per_class * w).sum() / (w.sum() + self.eps)
        else:
            loss = loss_per_class.mean()

        #print("====== loss.dtype:", loss.dtype)

        return loss







def clear_tensorboard_logs(log_dir="./tensorboard"):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"======= 清除旧的 TensorBoard 日志: {log_dir}")

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)







def plot_loss_acc_curves(history, num_class, save_dir="../fig/"):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss 曲线 ---
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(save_dir, "loss.jpg"))
    plt.close()

    # --- Accuracy 曲线 ---
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig(os.path.join(save_dir, "accuracy.jpg"))
    plt.close()

    # --- 每类准确率 (所有类放到一个大图中) ---
    ncols = 4  # 每行子图个数，可调整
    nrows = (num_class + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=False)
    # 如果只有一行，需要保证 axes 是二维
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]
    for c in range(num_class):
        row, col = divmod(c, ncols)
        ax = axes[row][col]
        ax.plot(epochs, [acc[c] for acc in history["train_per_class_acc"]], label="Train")
        ax.plot(epochs, [acc[c] for acc in history["val_per_class_acc"]], label="Val")
        ax.set_title(f"Class {c}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    # 删除多余的空子图
    for i in range(num_class, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.delaxes(axes[row][col])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_accuracy.jpg"), dpi=300, bbox_inches="tight")
    plt.close()


    # --- F1-score 曲线 ---
    plt.figure()
    plt.plot(epochs, history["train_f1"], label="Train F1")
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.legend()
    plt.title("Training and Validation F1-score")
    plt.savefig(os.path.join(save_dir, "f1_score.jpg"))
    plt.close()

    # --- 每类 F1-score ---
    ncols = 4
    nrows = (num_class + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=False)
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    for c in range(num_class):
        row, col = divmod(c, ncols)
        ax = axes[row][col]
        ax.plot(epochs, [f1[c] for f1 in history["train_per_class_f1"]], label="Train")
        ax.plot(epochs, [f1[c] for f1 in history["val_per_class_f1"]], label="Val")
        ax.set_title(f"Class {c} F1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1-score")
        ax.legend()
    for i in range(num_class, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.delaxes(axes[row][col])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_f1_score.jpg"), dpi=300, bbox_inches="tight")
    plt.close()


def convert_to_serializable(obj):
    """递归将 numpy 数据转换为 list，使其可 JSON 序列化"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    elif hasattr(obj, "tolist"):  # numpy.ndarray, numpy scalar
        return obj.tolist()
    else:
        return obj