# หน้าที่: เลือกโมเดล คำนวณ loss และจัด train/dev/test ด้วย Lightning
# ใช้กับ: ทุกชุดที่ใช้ DLModel

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from models.EEGDeformer import Deformer
from models.EEGNet import eegNet
from models.TSception import TSception
from models.EEGViT import EEGViT
from models.conformer import Conformer
import torchmetrics
import math
from models.l2sp import L2SP


# CB-Focal/CB-CE: สร้าง class weights จากจำนวนตัวอย่าง train และ beta
def effective_number_class_weights(class_counts, beta):
    """Cui et al. class weights, normalized to sum to num_classes.

    Uses expm1/log1p-style arithmetic to remain stable when beta is close to 1
    and class counts are large (as in windowed TUSZ).
    """
    counts = [int(count) for count in class_counts]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError(f'class_counts must contain positive integers, got {counts}')
    beta = float(beta)
    if not 0.0 < beta < 1.0:
        raise ValueError(f'cb_beta must be in (0, 1), got {beta}')

    log_beta = math.log(beta)
    effective_denominators = [-math.expm1(count * log_beta) for count in counts]
    raw_weights = [(1.0 - beta) / denominator for denominator in effective_denominators]
    scale = len(raw_weights) / sum(raw_weights)
    return torch.tensor(
        [weight * scale for weight in raw_weights], dtype=torch.float32
    )

# gamma0 คือ CE; gamma>0 ลดน้ำหนักตัวอย่างง่าย และใส่ class weight แยกจาก p_t
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha = per-class weight (เช่น [w_neg, w_pos]) — register เป็น buffer
        # เพื่อให้ย้ายขึ้น GPU พร้อมโมเดลอัตโนมัติ (ไม่งั้น weight อยู่ CPU แล้ว cross_entropy พัง)
        if alpha is not None:
            self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # Compute p_t from the unweighted CE.  If class weights are folded into CE
        # first, exp(-CE) becomes p_t ** alpha_t and changes the focal modulation.
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.numel() != inputs.shape[-1]:
                raise ValueError(
                    f"alpha has {self.alpha.numel()} entries, but inputs have "
                    f"{inputs.shape[-1]} classes"
                )
            alpha_t = self.alpha.to(dtype=inputs.dtype).gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# จุดรวมเลือก architecture; A4 ใช้ Deformer เดิมแต่เปลี่ยน hyperparameters
def init_model(args):
    model = None

    if args.model == 'Deformer':
        model = Deformer(
            num_chan=args.num_chan, num_time=args.num_time, temporal_kernel=args.kernel_length, num_kernel=args.T,
            num_classes=args.num_class, depth=int(args.num_layers-2), heads=args.AT,
            mlp_dim=args.AT, dim_head=args.AT, dropout=args.dropout)





    #### baselines ####
    if args.model == 'TSception':
        model = TSception(
            num_classes=args.num_class, input_size=(1, args.num_chan, args.num_time),
            sampling_rate=args.sampling_rate, num_T=15, num_S=15,
            hidden=32, dropout_rate=args.dropout)
    if args.model == 'EEGNet':
        model = eegNet(
            nChan=args.num_chan, nTime=args.num_time, nClass=args.num_class,
            dropoutP=args.dropout, F1=8, D=2,
            C1=64)

    if args.model == 'Conformer':
        if args.num_time == 384:
            n_hidden = 800
        if args.num_time == 800:
            n_hidden = 1880
        if args.num_time == 2000:
            n_hidden = 5080
        model = Conformer(n_classes=args.num_class, n_chan=args.num_chan, n_hidden=n_hidden)


    if args.model == 'EEGViT':
        if args.num_time == 384:
            n_patch = 24
        if args.num_time == 800:
            n_patch = 40
        if args.num_time == 2000:
            n_patch = 40
        model = EEGViT(
            num_chan=args.num_chan, num_time=args.num_time, num_patches=n_patch,
            num_classes=args.num_class)

    if model is None:
        raise ValueError(f"Unsupported model: {args.model}. Available: Deformer, EEGNet, TSception, Conformer, EEGViT")
    return model


def split_backbone_head_parameters(net):
    """Return disjoint Deformer backbone and classifier-head parameter lists."""
    backbone, head = [], []
    for name, parameter in net.named_parameters():
        (head if name.startswith("mlp_head.") else backbone).append(parameter)
    if not backbone or not head:
        raise ValueError(
            f"Gradual unfreezing requires both backbone and mlp_head parameters; "
            f"got backbone={len(backbone)}, head={len(head)}"
        )
    return backbone, head


def set_backbone_trainable(net, trainable):
    backbone, _ = split_backbone_head_parameters(net)
    for parameter in backbone:
        parameter.requires_grad = bool(trainable)


# Surgical fine-tuning (Lee et al., ICLR 2023): เลือกเทรนเฉพาะบางบล็อก
# ลำดับตาม forward: cnn_encoder -> pos_embedding -> transformer -> mlp_head
DEFORMER_PARAM_GROUPS = ("cnn_encoder", "pos_embedding", "transformer", "mlp_head")


def _in_group(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def split_parameter_groups(net, group_prefixes=DEFORMER_PARAM_GROUPS):
    """Partition named parameters into ordered groups by name prefix.

    Every parameter must land in exactly one group: an unmatched name means the
    architecture grew a module the surgical split does not know about, which
    would silently train or freeze it.
    """
    groups = {prefix: [] for prefix in group_prefixes}
    unmatched = []
    for name, parameter in net.named_parameters():
        for prefix in group_prefixes:
            if _in_group(name, prefix):
                groups[prefix].append(parameter)
                break
        else:
            unmatched.append(name)
    if unmatched:
        raise ValueError(
            f"parameters outside the surgical groups {list(group_prefixes)}: {unmatched}"
        )
    empty = [prefix for prefix, params in groups.items() if not params]
    if empty:
        raise ValueError(f"surgical groups with no parameters: {empty}")
    return groups


def apply_surgical_freeze(net, trainable_groups, group_prefixes=DEFORMER_PARAM_GROUPS):
    """Keep only `trainable_groups` trainable; freeze every other group."""
    groups = split_parameter_groups(net, group_prefixes)
    unknown = [g for g in trainable_groups if g not in groups]
    if unknown:
        raise ValueError(f"unknown surgical group(s) {unknown}; expected {list(groups)}")
    for prefix, parameters in groups.items():
        for parameter in parameters:
            parameter.requires_grad = prefix in trainable_groups
    return [prefix for prefix in group_prefixes if prefix not in trainable_groups]


def set_group_module_mode(net, prefixes, training):
    """train()/eval() every submodule under the given prefixes.

    Freezing parameters alone would still let a frozen block's BatchNorm update
    its running statistics, which is the same defect that affected the earlier
    gradual-unfreeze path.
    """
    for name, module in net.named_modules():
        if name and any(_in_group(name, prefix) for prefix in prefixes):
            module.train(training)


# training loop ร่วมของ fixed split และ LOSO; แต่ละ runner จัดแบ่งข้อมูลเอง
class DLModel(pl.LightningModule):
    def __init__(self, config):
        super(DLModel, self).__init__()
        self.save_hyperparameters()
        self.net = init_model(config)
        # A metric object owns state across all batches in one stage.  Separate
        # objects prevent train/validation/test state from leaking into each other.
        metric_kwargs = dict(task='multiclass', num_classes=config.num_class)
        self.train_acc = torchmetrics.Accuracy(**metric_kwargs)
        self.train_f1 = torchmetrics.F1Score(**metric_kwargs, average='macro')
        self.val_acc = torchmetrics.Accuracy(**metric_kwargs)
        self.val_f1 = torchmetrics.F1Score(**metric_kwargs, average='macro')
        self.test_acc = torchmetrics.Accuracy(**metric_kwargs)
        self.test_f1 = torchmetrics.F1Score(**metric_kwargs, average='macro')
        self.config = config
        self.l2sp_lambda = float(getattr(config, 'l2sp_lambda', 0.0))
        if not math.isfinite(self.l2sp_lambda) or self.l2sp_lambda < 0:
            raise ValueError('l2sp_lambda must be finite and >= 0')
        self.l2sp = L2SP(self.net) if self.l2sp_lambda > 0 else None

        self.val_probs = []
        self.val_labels = []
        self.register_buffer("optimal_threshold", torch.tensor(0.5))

        # [NEW] Focal Loss for Class Imbalance
        # focal_alpha > 0 = ใส่ class weight [1.0, w_pos] (rebalance โดยไม่ซ้ำข้อมูล → ปลอดภัยกว่า sampler)
        # 0 = ปิด (focal gamma อย่างเดียว, พฤติกรรมเดิม)
        fa = float(getattr(config, 'focal_alpha', 0.0))
        cb_beta = float(getattr(config, 'cb_beta', 0.0))
        focal_gamma = float(getattr(config, 'focal_gamma', 2.0))
        if cb_beta > 0:
            class_counts = getattr(config, 'cb_class_counts', None)
            if class_counts is None:
                raise ValueError('cb_beta > 0 requires train-derived cb_class_counts')
            alpha = effective_number_class_weights(class_counts, cb_beta)
            print(
                f'[CB Loss] beta={cb_beta:.8f}, gamma={focal_gamma:g}, '
                f'class_counts={list(class_counts)}, weights={alpha.tolist()}, '
                f'positive:negative={float(alpha[1] / alpha[0]):.6f}'
            )
        else:
            alpha = [1.0, fa] if fa > 0 else None
        self.criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)

        self.head_warmup_epochs = int(getattr(config, "head_warmup_epochs", 0))
        self.backbone_lr_scale = float(getattr(config, "backbone_lr_scale", 1.0))
        self._backbone_trainable = self.head_warmup_epochs == 0
        if not self._backbone_trainable:
            set_backbone_trainable(self.net, False)
            print(
                f"[GradualUnfreeze] backbone frozen for epochs "
                f"0..{self.head_warmup_epochs - 1}"
            )


        # Surgical fine-tuning: empty tuple keeps the historical behaviour of
        # training every parameter, so configurations that do not set it are
        # untouched.
        surgical = getattr(config, "surgical_train", None) or ()
        if isinstance(surgical, str):
            surgical = tuple(g.strip() for g in surgical.split(",") if g.strip())
        self.surgical_train = tuple(surgical)
        self._surgical_frozen = ()
        if self.surgical_train:
            if self.head_warmup_epochs > 0:
                raise ValueError(
                    "surgical_train and head_warmup_epochs both control freezing; "
                    "enable only one"
                )
            if self.l2sp is not None:
                # The surgical optimizer branch does not build the separate
                # backbone/head weight-decay groups that L2-SP relies on.
                raise ValueError("surgical_train cannot be combined with l2sp_lambda > 0")
            self._surgical_frozen = tuple(
                apply_surgical_freeze(self.net, self.surgical_train)
            )
            print(
                f"[Surgical] training {list(self.surgical_train)}; "
                f"frozen {list(self._surgical_frozen)}"
            )

    def forward(self, x):
        return self.net(x)

    # D2: จัดช่วง freeze backbone และช่วงเริ่ม fine-tune
    def on_train_epoch_start(self):
        if self._surgical_frozen:
            # Lightning puts the whole module back in train mode before this
            # hook, so the frozen blocks have to be returned to eval each epoch.
            set_group_module_mode(self.net, self._surgical_frozen, False)

        if not self._backbone_trainable:
            # Freezing parameters alone would still update backbone BatchNorm stats.
            self.net.eval()
            self.net.mlp_head.train()

        if (
            not self._backbone_trainable
            and self.current_epoch >= self.head_warmup_epochs
        ):
            set_backbone_trainable(self.net, True)
            self._backbone_trainable = True
            # The eval()/mlp_head.train() split above just ran for this same
            # call (self._backbone_trainable was still False), leaving the
            # backbone's BatchNorm/Dropout in eval mode. Without this, the
            # transition epoch would fine-tune the backbone with gradients
            # flowing while BN running stats are frozen and dropout is off.
            self.net.train()
            print(
                f"[GradualUnfreeze] backbone unfrozen at epoch {self.current_epoch}; "
                f"lr_scale={self.backbone_lr_scale:g}"
            )


    # คำนวณ classification loss และเพิ่ม L2-SP เฉพาะเมื่อเปิดใช้
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        # [OLD] Cross-Entropy (เอาไว้ใช้เป็น Baseline ปกติ)
        # loss = F.cross_entropy(y_hat, y)

        # [NEW] Focal Loss
        loss = self.criterion(y_hat, y)
        self.train_acc.update(y_hat, y)
        if self.l2sp is not None:
            penalty = self.l2sp(self.net)
            regularization = 0.5 * self.l2sp_lambda * penalty
            self.log('train_task_loss', loss, on_step=False, on_epoch=True)
            self.log('train_l2sp_penalty', penalty, on_step=False, on_epoch=True)
            self.log('train_l2sp_loss', regularization, on_step=False, on_epoch=True)
            loss = loss + regularization
        self.train_f1.update(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)

        if self.config.num_class == 2:
            probs = torch.softmax(y_hat, dim=1)[:, 1]
            self.val_probs.extend(probs.detach().cpu().numpy())
            self.val_labels.extend(y.detach().cpu().numpy())

        # [OLD] Cross-Entropy
        # loss = F.cross_entropy(y_hat, y)

        # [NEW] Focal Loss
        loss = self.criterion(y_hat, y)
        self.val_acc.update(y_hat, y)
        self.val_f1.update(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # dev AUPRC ใช้เลือก epoch; dev seizure-F1 ใช้เลือก threshold
    def on_validation_epoch_end(self):
        if self.config.num_class == 2 and len(self.val_probs) > 0:
            import numpy as np
            from sklearn.metrics import precision_recall_curve, average_precision_score

            # [NEW] AUPRC — threshold-free, โฟกัสคลาส seizure (ใช้เลือก checkpoint)
            # ต่างจาก val_f1 ที่เป็น macro@0.5 ซึ่งถูก background เจือจาง + ไม่ตรงกับ threshold ที่ deploy จริง
            auprc = average_precision_score(self.val_labels, self.val_probs)
            self.log("val_auprc", auprc, prog_bar=True, logger=True)

            precisions, recalls, thresholds = precision_recall_curve(self.val_labels, self.val_probs)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)

            if best_idx < len(thresholds):
                best_thresh = float(thresholds[best_idx])
                self.optimal_threshold.fill_(best_thresh)

            self.log("val_optimal_threshold", self.optimal_threshold.item(), prog_bar=True)

            self.val_probs.clear()
            self.val_labels.clear()

    # ใช้ threshold ที่ติดมากับ checkpoint; test_f1 ตรงนี้เป็น macro-F1
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # [OLD] Cross-Entropy
        # loss = F.cross_entropy(y_hat, y)

        # [NEW] Focal Loss
        loss = self.criterion(y_hat, y)
        if self.config.num_class == 2:
            probs = torch.softmax(y_hat, dim=1)[:, 1]
            preds = (probs >= self.optimal_threshold.item()).long()
        else:
            preds = y_hat

        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # Adam + cosine; D2/L2-SP อาจแยก LR หรือ weight decay ระหว่าง backbone/head
    def configure_optimizers(self):
        wd = getattr(self.config, "weight_decay", 1e-5)
        lr = float(self.config.lr)
        backbone_wd = getattr(self.config, 'backbone_weight_decay', None)
        if backbone_wd is None:
            backbone_wd = 0.0 if self.l2sp is not None else wd
        backbone_wd = float(backbone_wd)
        if not math.isfinite(backbone_wd) or backbone_wd < 0:
            raise ValueError('backbone_weight_decay must be finite and >= 0')
        if self.surgical_train:
            # Hand Adam only the unfrozen blocks so it keeps no optimizer state
            # for parameters that never receive a gradient.
            trainable = [p for p in self.net.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.trainer.max_epochs, 0
            )
            print(f"[Surgical] optimizer over {len(trainable)} trainable tensors, lr={lr:g}")
            return [optimizer], [scheduler]
        if self.l2sp is not None or backbone_wd != wd or self.head_warmup_epochs > 0 or self.backbone_lr_scale != 1.0:
            backbone, head = split_backbone_head_parameters(self.net)
            optimizer = torch.optim.Adam(
                [
                    {"params": backbone, "lr": lr * self.backbone_lr_scale, "name": "backbone", "weight_decay": backbone_wd},
                    {"params": head, "lr": lr, "name": "head"},
                ],
                weight_decay=wd,
            )
            print(
                f"[GradualUnfreeze] optimizer backbone_lr="
                f"{lr * self.backbone_lr_scale:g}, head_lr={lr:g}"
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        )
        return [optimizer], [scheduler]
