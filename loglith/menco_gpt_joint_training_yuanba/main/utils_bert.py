import torch
import os
import json
import random 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score  
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import time
import shutil
import torch.nn.functional as F


from utils import chunkify_tokens, load_checkpoint_for_resume, compute_per_class_accuracy, plot_loss_acc_curves
from utils import FocalTverskyLossBLC, FocalLoss, save_checkpoint, clear_tensorboard_logs, convert_to_serializable
print("======= 在预训练和微调的for循环中更新了 每epoch的种子!!!")









def pretrain_network_bert(args, model, train_loader, val_loader, device,
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
    history = {"train_loss": [], "val_loss": [],
               "train_mask_ratio_actual": [], "val_mask_ratio_actual": []}

    stime = time.time()
    for ep in range(1, args.epochs + 1):
        np.random.seed(42 + ep)
        torch.manual_seed(42 + ep)
        torch.cuda.manual_seed_all(42 + ep)

        # ---------------------- Training ----------------------
        model.train()
        total_loss = 0.0
        train_mask_ratio_sum = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            xb_ori, masko = batch["fea_log"], batch["masko"]  # (B, T=512, C)

            # chunk 化
            x_tokens, mask_tokens = chunkify_tokens(xb_ori, masko, args)  

            optimizer.zero_grad()
            # 预训练任务：重建  曲线
            loss, recon, bert_mask, info = model.pretrain_step(x_tokens, mask_tokens)  # (B,64,5)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_mask_ratio_sum += info["mask_ratio_actual"]

        train_loss = total_loss / len(train_loader)
        history["train_loss"].append(train_loss)
        history["train_mask_ratio_actual"].append(train_mask_ratio_sum / len(train_loader))
        
        
        # ---------------------- Validation ----------------------
        model.eval()
        val_total = 0.0
        val_mask_ratio_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                xb_ori, masko = batch["fea_log"], batch["masko"]

                x_tokens, mask_tokens = chunkify_tokens(xb_ori, masko, args)  
                
                loss, recon, bert_mask, info = model.pretrain_step(x_tokens, mask_tokens)

                val_total += loss.item()
                val_mask_ratio_sum += info["mask_ratio_actual"]

        val_loss = val_total / len(val_loader)
        history["val_loss"].append(val_loss)
        history["val_mask_ratio_actual"].append(val_mask_ratio_sum / len(val_loader))

        # ---------------------- Logging ----------------------
        etime = time.time()
        print(f"[Pretrain {experiment_name}] Epoch {ep}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {etime-stime:.2f}s")

        writer.add_scalar("Loss/Train", train_loss, ep)
        writer.add_scalar("Loss/Val", val_loss, ep)
        writer.add_scalar("MaskRatio/Train", history["train_mask_ratio_actual"][-1], ep)
        writer.add_scalar("MaskRatio/Val", history["val_mask_ratio_actual"][-1], ep)

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



def train_network_bert(args, model, train_loader, val_loader, num_class, 
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
                                        alpha=0.3, beta=0.7, gamma=0.75, 
                                        eps=1e-6,
                                        class_weights=None,     # 可选: (C,)
                                        ignore_index=None       # 可选: 忽略某些标签
                                        )
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")


    ################# 下面这部分是不同的微调策略，以及加载断点继续学习  #################
    # ####微调. 不冻结，但弱化重构学习率
    # encoder_params = []
    # head_params = []
    # print("====== ⚠️ 微调：不冻结，但弱化重构学习率, 一定注意模型架构变化，这里微调的架构可能需要随之变化")
    # for name, p in model.named_parameters():
    #     if "encoder" in name or "input_proj" in name:
    #         encoder_params.append(p)
    #     else:
    #         head_params.append(p)

    # optimizer = torch.optim.Adam([
    #     {"params": encoder_params, "lr": args.main_net_lr},   # 微调主网络的学习率
    #     {"params": head_params, "lr": args.lr}
    # ])


    ####下述是部分微调及其断点继续学习的代码
    # =========================================================
    # 1) 冻结预训练模块，只训练微调模块（head）
    # =========================================================
    freeze_keywords = ("encoder", "input_proj")  # 按你模型命名改
    print(f"====== ⚠️ 微调：冻结预训练模块 {freeze_keywords}，仅训练其余参数(微调头)")
    freeze_pretrain_unfreeze_head(model, freeze_keywords=freeze_keywords)

    # 只把 requires_grad=True 的参数交给 optimizer（单一学习率）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)


    # =========================================================
    # 2) 断点加载：预训练权重/微调断点 二合一
    # =========================================================
    if hasattr(args, "resume_ckpt") and args.resume_ckpt is not None and os.path.exists(args.resume_ckpt):
        print(f"====== ✅ Resuming from checkpoint: {args.resume_ckpt}")
        resume_epoch = load_checkpoint_for_resume(args.resume_ckpt, model, optimizer, device=device)
        print(f"====== ✅ resume_epoch={resume_epoch} (仅用于记录，不影响你的for循环)")

        # 只有当真的加载了 optimizer_state_dict 时，state 才可能非空；空的话循环也没影响
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        print("====== args.resume_ckpt", getattr(args, "resume_ckpt", None))
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
        train_loss, train_acc, train_per_class_acc, train_f1, train_per_class_f1 = train_one_epoch_bert(
            args, model, train_loader, criterion, optimizer, device, num_class, lambda_pre=lambda_pre,
        )

        # train_time = time.time()-atime
        # print(f"训练消耗的时间为:{train_time}")

        val_loss, val_acc, val_per_class_acc, val_f1, val_per_class_f1 = validate_one_epoch_bert(
            args, model, val_loader, criterion, device, num_class, lambda_pre=lambda_pre,
        )


        
        # val_time = time.time()-atime-train_time
        # print(f"验证消耗的时间为:{val_time}")

        # 保存历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_per_class_acc"].append(train_per_class_acc)
        history["val_per_class_acc"].append(val_per_class_acc)

        # ===== 新增 F1 =====
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_per_class_f1"].append(train_per_class_f1)
        history["val_per_class_f1"].append(val_per_class_f1)


        # TensorBoard 可视化
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)   
        writer.add_scalar("F1/Val", val_f1, epoch)

        for c in range(num_class):
            writer.add_scalar(f"ClassAcc/Train_class_{c}", train_per_class_acc[c], epoch)
            writer.add_scalar(f"ClassAcc/Val_class_{c}", val_per_class_acc[c], epoch)

        etime = time.time()

        # 打印信息
        #if epoch % epoch_inte == 0 or epoch == 1:
        print(f"[{experiment_name} Epoch {epoch}/{args.epochs}] | "
                f"Train: total={train_loss:.4f} acc={train_acc:.4f} || "
                f"Val: total={val_loss:.4f} acc={val_acc:.4f} || "
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










def train_one_epoch_bert(args, model, train_loader, criterion, optimizer, device, num_class,
                    lambda_pre=None):
    model.train()

    total_loss_sum = 0.0

    train_correct, train_total = 0, 0
    all_preds, all_targets = [], []

    for data in train_loader:
        data = {key: tensor.to(device) for key, tensor in data.items()}
        batch_x, target, masko = data['fea_log'], data['lith'], data['masko']   

        # ===== chunkify =====
        x_tokens, mask_tokens = chunkify_tokens(batch_x, masko, args)        

        optimizer.zero_grad()
        outputs = model.finetune_step(x_tokens, mask_tokens)               

        loss = criterion(outputs.view(-1, num_class), target.view(-1))

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            print("====== grad_norm nan/inf -> skip this batch (no optimizer.step)")
            optimizer.zero_grad(set_to_none=True)
            continue   # ✅关键：跳过 optimizer.step()

        optimizer.step()

        total_loss_sum += loss.item()
       
        preds = outputs.argmax(dim=-1)  
        train_correct += (preds == target).sum().item()
        train_total += target.numel()

        all_preds.append(preds.detach())
        all_targets.append(target.detach())


    # ===== epoch stats =====
    avg_total_loss = total_loss_sum / len(train_loader)

    acc = train_correct / train_total
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

    # ⭐ 返回时把分类loss和重构loss也返回
    return avg_total_loss, acc, per_class_acc, f1_macro, per_class_f1



def validate_one_epoch_bert(args, model, val_loader, criterion, device, num_class,
                       lambda_pre=None):
    model.eval()

    total_loss_sum = 0.0

    val_correct, val_total = 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data in val_loader:
            data = {key: tensor.to(device) for key, tensor in data.items()}
            batch_x, target, masko = data['fea_log'], data['lith'], data['masko']

            x_tokens, mask_tokens = chunkify_tokens(batch_x, masko, args)

            outputs = model.finetune_step(x_tokens, mask_tokens)  

            loss = criterion(outputs.view(-1, num_class), target.view(-1))

            total_loss_sum += loss.item()

            preds = outputs.argmax(dim=-1)
            val_correct += (preds == target).sum().item()
            val_total += target.numel()

            all_preds.append(preds)
            all_targets.append(target)

    avg_total_loss = total_loss_sum / len(val_loader)

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

    return avg_total_loss, acc, per_class_acc, f1_macro, per_class_f1






def set_requires_grad_by_name(model, freeze_keywords=None, freeze=True):
    """
    freeze=True:  冻结命中关键词的参数
    freeze=False: 解冻命中关键词的参数
    """
    for name, p in model.named_parameters():
        hit = any(k in name for k in freeze_keywords)
        if hit:
            p.requires_grad = (not freeze)

def freeze_pretrain_unfreeze_head(model, freeze_keywords=None):
    """
    冻结预训练模块(命中关键词)，解冻其它作为微调头。
    """
    # 先全部解冻，再按规则冻结，更不容易漏
    for _, p in model.named_parameters():
        p.requires_grad = True

    # 冻结预训练模块
    set_requires_grad_by_name(model, freeze_keywords=freeze_keywords, freeze=True)

    # 保证 head（非预训练模块）是可训练
    #（因为我们上面先全 True，再冻结命中关键词的，所以这里其实不用再显式解冻）
