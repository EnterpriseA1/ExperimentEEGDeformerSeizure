# หน้าที่: ทดสอบความถูกต้องของ D1/D2 backbone_strict
# ใช้กับ: D1/D2 backbone_strict

import json

import torch
from torch import nn

from training.task import initialize_model_from_checkpoint


class _TinyTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.encoder = nn.Linear(3, 2)
        self.net.mlp_head = nn.Sequential(nn.Linear(2, 2))
        self.register_buffer("optimal_threshold", torch.tensor(0.5))


def test_backbone_strict_loads_encoder_and_leaves_head_random(tmp_path):
    source = _TinyTransferModel()
    target = _TinyTransferModel()
    with torch.no_grad():
        source.net.encoder.weight.fill_(3.0)
        source.net.encoder.bias.fill_(4.0)
        source.net.mlp_head[0].weight.fill_(5.0)
        source.net.mlp_head[0].bias.fill_(6.0)
        source.optimal_threshold.fill_(0.9)
        target.net.encoder.weight.fill_(-3.0)
        target.net.encoder.bias.fill_(-4.0)
        target.net.mlp_head[0].weight.fill_(-5.0)
        target.net.mlp_head[0].bias.fill_(-6.0)

    checkpoint = tmp_path / "source.ckpt"
    audit_path = tmp_path / "initialization_audit.json"
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    initialize_model_from_checkpoint(
        model=target,
        checkpoint_path=str(checkpoint),
        policy="backbone_strict",
        audit_path=str(audit_path),
    )

    assert torch.equal(target.net.encoder.weight, source.net.encoder.weight)
    assert torch.equal(target.net.encoder.bias, source.net.encoder.bias)
    assert torch.all(target.net.mlp_head[0].weight == -5.0)
    assert torch.all(target.net.mlp_head[0].bias == -6.0)
    assert target.optimal_threshold.item() == 0.5

    audit = json.loads(audit_path.read_text())
    assert audit["policy"] == "backbone_strict"
    assert audit["loaded_count"] == 2
    assert audit["missing_network"] == []
    assert audit["skipped_shape"] == []
    assert audit["unexpected"] == []
    assert audit["source_threshold_loaded"] is False
    assert audit["optimizer_state_loaded"] is False
    assert audit["intentionally_excluded_network"] == [
        "net.mlp_head.0.bias",
        "net.mlp_head.0.weight",
    ]
