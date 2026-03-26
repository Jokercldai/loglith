import torch
import os
import json
import random 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score  
from scipy.special import softmax
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


# def pretrain_network(args, model, train_loader, val_loader, device,
#              epoch_inte=20):
#     """
#     自监督预训练阶段：重建掩码特征
#     """

#     ##清除旧的tensorboard
#     clear_tensorboard_logs(log_dir = '../tensorboard/pretrain/')
#     ##建立新的tensorboard
#     tensorboard_dir = os.path.join('../tensorboard/pretrain/')
#     if not os.path.exists(tensorboard_dir):
#         os.makedirs(tensorboard_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=tensorboard_dir)


#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     best_val_loss = float('inf')
#     best_model_path = os.path.join('../model/', 'best_pretrain_model.pth')

#     train_losses, val_losses = [], []

#     stime = time.time()
#     for ep in range(1, args.epochs + 1):
#         # 每个 epoch 改一个确定性的偏移种子  使得不同epoch之间的样本不同
#         np.random.seed(42 + ep)
#         torch.manual_seed(42 + ep)
#         torch.cuda.manual_seed_all(42 + ep)

#         model.train()
#         train_loss = 0.0
#         for data in train_loader:
#             data = {key: tensor.to(device) for key, tensor in data.items()}
#             xb_ori, masko = data['fea_log'], data['masko']
#             #print('====== xb.size()', xb.size())  #(b, sam_len, num_log)

#             xb = xb_ori[:, :-1, :]
#             tgt = xb_ori[:, 1:, :]
#             masko_tgt = masko[:, 1:, :]

#             optimizer.zero_grad()
#             out = model(xb, model_task=args.stage)
#             #print("====== out.size(), tgt.size()", out.size(), tgt.size())
            
#             loss = masked_mse_loss(out, tgt, masko_tgt)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         train_loss /= len(train_loader)
#         train_losses.append(train_loss)

#         # ------------------ Validation ------------------
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for data in val_loader:
#                 data = {key: tensor.to(device) for key, tensor in data.items()}
#                 xb_ori, masko = data['fea_log'], data['masko']

#                 xb = xb_ori[:, :-1, :]
#                 tgt = xb_ori[:, 1:, :]
#                 masko_tgt = masko[:, 1:, :]
              
#                 out = model(xb, model_task=args.stage)
#                 loss = masked_mse_loss(out, tgt, masko_tgt)
#                 val_loss += loss.item()
#         val_loss /= len(val_loader)
#         val_losses.append(val_loss)

#         etime = time.time()
#         print(f"[Pretrain] Epoch {ep}/{args.epochs} | "
#               f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
#               f"Time: {etime-stime:.2f}s")

#         #将loss写入TensorBoard
#         writer.add_scalar("Loss/Train", train_loss, ep)
#         writer.add_scalar("Loss/Validation", val_loss, ep)

#         # ------------------ Save best ------------------
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), best_model_path)
#             print(f"--> Saved best model (val_loss={val_loss:.4f})")
#             best_epo = ep

#         if ep % epoch_inte == 0:
#             torch.save(model.state_dict(), f"../model/pretrain_epoch_{ep}.pth")

#     # ------------------ Plot Loss Curve ------------------
#     plt.figure(figsize=(8,5))
#     plt.plot(train_losses, label='Train Loss', linewidth=2)
#     plt.plot(val_losses, label='Val Loss', linewidth=2)
#     plt.xlabel('Epoch')
#     plt.ylabel('MSE Loss')
#     plt.title('Pretraining Reconstruction Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('../fig/pretrain_loss.jpg', dpi=300)
#     plt.show()

#     writer.close()
#     print("======== Pretraining finished!!!")
#     print(f"======== 最好模型对应epoch= {best_epo} !")




def contrast_train_network(args, model, train_loader, val_loader, num_class, 
                  device=None, epoch_inte=20, freeze_encoder=True):
    
    #experiment_name = f"{args.stage}_{args.model_type}"
    experiment_name = args.experiment_name
    print(f"====== 当前实验：{experiment_name} ======")

    tensorboard_dir = f"../tensorboard/{experiment_name}/"
    ##清除旧的tensorboard
    clear_tensorboard_logs(log_dir = tensorboard_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

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
                                        alpha=0.3, beta=0.7, gamma=1.3, 
                                        eps=1e-6,
                                        class_weights=lith_weight,     # 可选: (C,)
                                        ignore_index=None       # 可选: 忽略某些标签
                                        )
        print("====== 注意查看gamma权重是否使用1.25～2.0的范围，这个范围通常对不平衡分类更有效，class_weight是否正确加载了lith_weight")


    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    

    ####微调 迁移学习
    ###model is expected to have classifier head; freeze encoder optionally
    #print("======== 全量参数微调！！")
    if freeze_encoder:
        print("====== 开启模型冻结！！！")
        for name, p in model.named_parameters():
            if 'classifier' not in name:
                p.requires_grad = False  #如果条件成立(即不是分类头参数)，则执行 p.requires_grad = False，表示这些参数在训练过程中不需要计算梯度，也就不会被更新

    #创建Adam优化器时，使用filter筛选出需要梯度更新的参数(requires_grad=True)，也可以不加filter的，依旧不会更新冻结的参数的
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"======= 可训练参数: {trainable_params:,} / 总参数: {total_params:,} "
        f"({trainable_params/total_params:.2%} 可训练)")


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
        "train_acc": [],
        "val_acc": [],
        "train_per_class_acc": [],
        "val_per_class_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_per_class_f1": [],
        "val_per_class_f1": []
    }
    stime = time.time()
    for epoch in range(1, args.epochs + 1):

        # 每个 epoch 改一个确定性的偏移种子  使得不同epoch之间的样本不同
        np.random.seed(42 + epoch)
        torch.manual_seed(42 + epoch)
        torch.cuda.manual_seed_all(42 + epoch)

        atime = time.time()
        train_loss, train_acc, train_per_class_acc, train_f1, train_per_class_f1  = train_one_epoch(
            args, model, train_loader, criterion, optimizer, device, 
            num_class,
        )

        # train_time = time.time()-atime
        # print(f"训练消耗的时间为:{train_time}")

        val_loss, val_acc, val_per_class_acc, val_f1, val_per_class_f1  = validate_one_epoch(
            args, model, val_loader, criterion, device, num_class,
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
        print(f"[{experiment_name} Epoch {epoch}/{args.epochs}] |"
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"Time: {etime-stime:.3f}")

        # 保存周期性模型
        if epoch % epoch_inte == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"epoch_{epoch}.pth"))


        if (val_loss < best_val_loss) or (np.isclose(val_loss, best_val_loss) and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint({
                #'epoch': epoch,
                'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc
            }, best_model_path)
            best_epoch = epoch  #标记一下最好的epoch
            print(f"--> New best model saved (epoch {epoch}) val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")  #[{experiment_name}] 

    writer.close()
    print(f"====== 训练完成，最优模型已保存到:{best_model_path}", "\n",
          f"====== 最好模型的epoch是:{best_epoch}")
    
    # 绘制并保存图像
    fig_dir = f"../fig/{experiment_name}/"
    plot_loss_acc_curves(history, num_class, save_dir=fig_dir)

    # ===== 保存 history 到 JSON =====
    history_dir = f"../history/{experiment_name}/"
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "training_history.json")
    history = convert_to_serializable(history)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    #print(f"====== 训练过程 history data 已保存到: {history_path}")







def train_one_epoch(args, model, train_loader, criterion, optimizer, device, num_class):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for data in train_loader:
        ##加载数据
            # batch_x: (bsize, seqlen, num_types)
            # target: (bsize, seqlen)
        data = {key: tensor.to(device) for key, tensor in data.items()}
        batch_x, target = data['fea_log'], data['lith']

        optimizer.zero_grad()
        # no masking in finetune, use full signals
        outputs = model(batch_x, model_task=args.stage)
        # print("====== outputs.size(), target.size()", outputs.size(), target.size())  
        # print("====== outputs.view(-1, num_class).size()", outputs.view(-1, num_class).size())
        # print("====== target.view(-1).size()", target.view(-1).size())
        loss = criterion(
                outputs.view(-1, num_class),   
                target.view(-1)                
            )
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=-1)  # (bsize, seqlen)
        # print("preds.size()", preds.size())
        # print("target.size()", target.size())  # (bsize, seqlen)
        train_correct += (preds == target).sum().item()
        train_total += target.numel()

        all_preds.append(preds.detach())
        all_targets.append(target.detach())

    avg_loss = train_loss / len(train_loader)
    acc = train_correct / train_total
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    ## 为节省时间这个就不算了先
    per_class_acc = compute_per_class_accuracy(all_preds, all_targets, num_class)

    # print("====== all_preds:", all_preds)
    # print("====== all_targets:", all_targets)
    # ===== 新增 F1-score =====
    f1_macro = f1_score(all_targets.view(-1).cpu().numpy(),
                        all_preds.view(-1).cpu().numpy(),
                        average='macro',   #macro  micro None
                        zero_division=0)  ##没有出现的岩性类别的F1-score置0或1
    per_class_f1 = f1_score(all_targets.view(-1).cpu().numpy(),
                            all_preds.view(-1).cpu().numpy(),
                            average=None,
                            labels=list(range(num_class)),
                            zero_division=0)
    return avg_loss, acc, per_class_acc, f1_macro, per_class_f1


def validate_one_epoch(args, model, val_loader, criterion, device, num_class):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data in val_loader:
            data = {key: tensor.to(device) for key, tensor in data.items()}
            batch_x, target = data['fea_log'], data['lith']

            # no masking in finetune, use full signals
            outputs = model(batch_x, args.stage)
            loss = criterion(outputs.view(-1, num_class), target.view(-1))
            val_loss += loss.item()

            preds = outputs.argmax(dim=-1)
            val_correct += (preds == target).sum().item()
            val_total += target.numel()

            all_preds.append(preds)
            all_targets.append(target)

    avg_loss = val_loss / len(val_loader)
    acc = val_correct / val_total
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    per_class_acc = compute_per_class_accuracy(all_preds, all_targets, num_class)

    # ===== 新增 F1-score =====
    f1_macro = f1_score(all_targets.view(-1).cpu().numpy(),
                        all_preds.view(-1).cpu().numpy(),
                        average='macro',
                        zero_division=0)
    per_class_f1 = f1_score(all_targets.view(-1).cpu().numpy(),
                            all_preds.view(-1).cpu().numpy(),
                            average=None,
                            labels=list(range(num_class)),
                            zero_division=0)

    return avg_loss, acc, per_class_acc, f1_macro, per_class_f1



def compute_per_class_accuracy(preds, targets, num_class):
    """计算每类的准确率"""
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