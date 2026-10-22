# หน้าที่: จัด data loaders, โหลดน้ำหนักตั้งต้น และควบคุมการเทรน/เลือก checkpoint
# ใช้กับ: แกน fixed split; initialization ใช้ร่วมกับ LOSO

from models.model_handler import DLModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import *
import json
import os






# Transfer: โหลดเฉพาะ network ตาม policy; ไม่ยก optimizer/threshold จาก source มาใช้
def initialize_model_from_checkpoint(model, checkpoint_path, policy, audit_path):
    """Initialize network weights without restoring optimizer or source threshold state."""
    if policy == 'none':
        return None
    if getattr(model, 'l2sp', None) is not None:
        if policy not in ('network_strict', 'backbone_strict'):
            raise ValueError('L2-SP requires strict source initialization')
        if bool(model.l2sp.ready):
            raise RuntimeError('L2-SP source reference already captured')
    if not checkpoint_path or not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f'Initialization checkpoint not found: {checkpoint_path}')
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    source_state = checkpoint.get('state_dict', checkpoint)
    source_network = {key: value for key, value in source_state.items() if key.startswith('net.')}
    target_state = model.state_dict()
    target_network_keys = {key for key in target_state if key.startswith('net.')}

    # Cross-domain transfer can retain the feature extractor while learning a fresh
    # target-domain decision boundary.  Deformer's classifier lives entirely under
    # net.mlp_head.*, so exclude it explicitly and audit every omitted tensor.
    excluded_network = []
    if policy == 'backbone_strict':
        head_prefix = 'net.mlp_head.'
        excluded_network = sorted(
            key for key in source_network if key.startswith(head_prefix)
        )
        if not excluded_network:
            raise ValueError(
                'backbone_strict expected source tensors under net.mlp_head.*'
            )
        source_network = {
            key: value for key, value in source_network.items()
            if not key.startswith(head_prefix)
        }
        target_network_keys = {
            key for key in target_network_keys if not key.startswith(head_prefix)
        }

    loaded = {}
    skipped_shape = []
    unexpected = []
    for key, value in source_network.items():
        if key not in target_state:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append({
                'name': key,
                'source_shape': list(value.shape),
                'target_shape': list(target_state[key].shape),
            })
        else:
            loaded[key] = value
    missing = sorted(target_network_keys - set(loaded))

    if policy in ('network_strict', 'backbone_strict') and (
        skipped_shape or unexpected or missing
    ):
        raise ValueError(
            f'{policy} initialization mismatch: '
            f'shape={skipped_shape}, unexpected={unexpected}, missing={missing}'
        )
    if policy not in ('network_strict', 'backbone_strict', 'compatible'):
        raise ValueError(f'Unknown initialization policy: {policy}')
    target_state.update(loaded)
    model.load_state_dict(target_state, strict=True)

    if getattr(model, 'l2sp', None) is not None:
        model.l2sp.capture(model.net)

    audit = {
        'checkpoint': osp.abspath(checkpoint_path),
        'policy': policy,
        'loaded_count': len(loaded),
        'loaded': sorted(loaded),
        'intentionally_excluded_network': excluded_network,
        'skipped_shape': skipped_shape,
        'unexpected': sorted(unexpected),
        'missing_network': missing,
        'source_threshold_loaded': False,
        'optimizer_state_loaded': False,
        'l2sp_reference_captured': getattr(model, 'l2sp', None) is not None,
    }
    os.makedirs(osp.dirname(audit_path), exist_ok=True)
    with open(audit_path, 'w', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    print(
        f'[Transfer Init] policy={policy}, loaded={len(loaded)}, '
        f'excluded={len(excluded_network)}, shape_skipped={len(skipped_shape)}, '
        f'missing={len(missing)}'
    )
    return audit


# Fixed split: train เรียนรู้ → dev เลือก checkpoint/threshold → eval วัดผลสุดท้าย
def TUSZ_train(args, experiment_ID, logs_name):
    if getattr(args, 'train_patient_balanced', False):
        raise ValueError('D3/D3.1 patient-balanced training has been removed from TUSZ_train')
    import glob
    import re
    import os.path as osp

    pl.seed_everything(seed=args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Load train dataset (all .pkl files inside data_eeg_TUSZ_seizure_train)
    train_load_path = osp.join(args.load_path, 'data_{}_{}_{}_train'.format(args.data_format, args.dataset, args.label_type))
    train_files = sorted(glob.glob(osp.join(train_load_path, "sub*.pkl")))
    train_idx = [int(re.search(r'sub(\d+)\.pkl', f).group(1)) for f in train_files]

    # Verify that train files exist
    if len(train_idx) == 0:
        raise ValueError(f"No processed train files found in {train_load_path}. Please run TUSZ.py for train split first.")

    data_train, label_train = load_data(load_path=train_load_path, load_idx=train_idx, concat=False)

    if float(getattr(args, 'cb_beta', 0.0)) > 0:
        train_labels = np.concatenate(label_train).reshape(-1)
        args.cb_class_counts = [
            int(np.sum(train_labels == class_id))
            for class_id in range(args.num_class)
        ]
        if any(count == 0 for count in args.cb_class_counts):
            raise ValueError(
                f'CB Loss requires every class in train split; got {args.cb_class_counts}'
            )
        print(f'[CB Loss] train-derived class counts: {args.cb_class_counts}')

    # 2. Load dev/validation dataset (all .pkl files inside data_eeg_TUSZ_seizure_dev)
    dev_load_path = osp.join(args.load_path, 'data_{}_{}_{}_dev'.format(args.data_format, args.dataset, args.label_type))
    dev_files = sorted(glob.glob(osp.join(dev_load_path, "sub*.pkl")))
    dev_idx = [int(re.search(r'sub(\d+)\.pkl', f).group(1)) for f in dev_files]

    # Verify that dev files exist
    if len(dev_idx) == 0:
        raise ValueError(f"No processed dev files found in {dev_load_path}. Please run TUSZ.py for dev split first.")

    data_val, label_val = load_data(load_path=dev_load_path, load_idx=dev_idx, concat=False)

    # 3. Load eval only when explicitly requested. Dev-gated pilots avoid eval peeking.
    data_eval = label_eval = None
    if not getattr(args, "skip_final_eval", False):
        eval_load_path = osp.join(args.load_path, "data_{}_{}_{}_eval".format(
            args.data_format, args.dataset, args.label_type
        ))
        eval_files = sorted(glob.glob(osp.join(eval_load_path, "sub*.pkl")))
        eval_idx = [int(re.search(r"sub(\d+)\.pkl", f).group(1)) for f in eval_files]
        if len(eval_idx) == 0:
            raise ValueError(
                f"No processed eval files found in {eval_load_path}; "
                "refusing to fall back to dev for final evaluation"
            )
        data_eval, label_eval = load_data(
            load_path=eval_load_path, load_idx=eval_idx, concat=False
        )

    def get_total_size(lst):
        return sum(d.shape[0] for d in lst)

    eval_size = 'skipped' if data_eval is None else get_total_size(data_eval)
    print('Final Sizes -> Train:{} Val:{} Eval:{}'.format(
        get_total_size(data_train), get_total_size(data_val), eval_size
    ))

    # Reorder the data channel info
    idx, _ = get_selected_channel_info(
        load_path=train_load_path,
        graph_type=args.graph_type,
        channel_subset=getattr(args, 'channel_subset', 'all'),
    )
    if len(idx) != args.num_chan:
        raise ValueError(
            f'Input channel count mismatch: selected {len(idx)} channels but '
            f'--num-chan={args.num_chan}'
        )

    # Prepare dataloaders
    # norm_mode ต้องเหมือนกันทั้ง 3 split ไม่งั้น train/test เห็นสเกลคนละแบบ
    norm_mode = canonical_norm_mode(getattr(args, 'norm_mode', LEGACY_NORM_MODE))
    norm_clip = getattr(args, 'norm_clip', 0.0)
    norm_stats = None
    if norm_mode == TRAIN_CHANNEL_F32_NORM_MODE:
        mean, std, sample_count = compute_train_channel_stats(data_train, idx_channels=idx)
        norm_stats = (mean, std)
        stats_path = args.norm_stats_path
        ensure_path(osp.dirname(stats_path))
        np.savez(
            stats_path,
            mean=mean,
            std=std,
            scalar_samples_per_channel=np.asarray(sample_count, dtype=np.int64),
            channel_indices=np.asarray(idx, dtype=np.int64),
            norm_mode=np.asarray(norm_mode),
            input_cast_dtype=np.asarray('float32'),
            runtime_dtype=np.asarray('float32'),
            denominator_floor=np.asarray(float(TRAIN_CHANNEL_F32_EPS), dtype=np.float32),
        )
        print(
            f'[Normalization] {norm_mode}: train-only stats saved to {stats_path}; '
            f'samples/channel={sample_count}, std range=({std.min():.3e}, {std.max():.3e})'
        )
    train_loader = prepare_data_for_training(
        data_list=data_train, label_list=label_train, idx=idx,
        batch_size=args.batch_size, shuffle=True,
        use_weighted_sampler=getattr(args, 'use_weighted_sampler', False),
        norm_mode=norm_mode, norm_clip=norm_clip,
        train_neg_pos_ratio=getattr(args, 'train_neg_pos_ratio', 0.0),
        sampler_seed=args.random_seed,
        norm_stats=norm_stats,
    )
    val_loader = prepare_data_for_training(data_list=data_val, label_list=label_val, idx=idx, batch_size=args.batch_size, shuffle=False, norm_mode=norm_mode, norm_clip=norm_clip, norm_stats=norm_stats)
    eval_loader = None
    if data_eval is not None:
        eval_loader = prepare_data_for_training(
            data_list=data_eval, label_list=label_eval, idx=idx,
            batch_size=args.batch_size, shuffle=False, norm_mode=norm_mode,
            norm_clip=norm_clip, norm_stats=norm_stats,
        )

    # Train and test the model
    model = DLModel(config=args)
    if getattr(args, 'init_policy', 'none') != 'none':
        initialize_model_from_checkpoint(
            model=model,
            checkpoint_path=args.init_checkpoint,
            policy=args.init_policy,
            audit_path=osp.join(
                args.save_path, logs_name, experiment_ID, 'initialization_audit.json'
            ),
        )
    # เลือก checkpoint ด้วย AUPRC (threshold-free, โฟกัส seizure) แทน val_f1 (macro@0.5)
    checkpoint_callback = ModelCheckpoint(monitor="val_auprc", mode='max')
    ensure_path(args.save_path)
    # TensorBoard (สำหรับ visualize) + CSV (metrics.csv: train/val/test loss รายเอพอค อ่านง่าย)
    logger = [
        TensorBoardLogger(save_dir=args.save_path, version=experiment_ID, name=logs_name),
        CSVLogger(save_dir=args.save_path, version=experiment_ID, name=logs_name),
    ]

    if args.mixed_precision:
        trainer = pl.Trainer(
            accelerator="gpu", devices=[args.gpu], max_epochs=args.max_epoch, logger=logger,
            callbacks=[checkpoint_callback], precision='16-mixed'
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu", devices=[args.gpu], max_epochs=args.max_epoch, logger=logger,
            callbacks=[checkpoint_callback]
        )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_val_metrics = trainer.checkpoint_callback.best_model_score.item()

    if getattr(args, "skip_final_eval", False):
        print("[Dev Gate] Held-out eval skipped by --skip-final-eval")
        return [{"best_val": best_val_metrics}]

    # Run final test on the true held-out eval set using the best checkpoint.
    results = trainer.test(ckpt_path="best", dataloaders=eval_loader)
    results[0]["best_val"] = best_val_metrics
    return results
