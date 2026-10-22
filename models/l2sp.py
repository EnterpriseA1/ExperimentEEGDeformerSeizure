# หน้าที่: วัดระยะห่างของ backbone ปัจจุบันจากน้ำหนัก source เพื่อใช้ regularization
# ใช้กับ: L2-SP transfer LOSO

"""L2-SP: float32 squared distance to a fixed, checkpointed source backbone."""
import torch
from torch import nn


class L2SP(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.names = tuple(n for n, _ in network.named_parameters()
                           if not n.startswith('mlp_head.'))
        if not self.names or not hasattr(network, 'mlp_head'):
            raise ValueError('L2-SP requires a backbone and mlp_head classifier')
        parameters = dict(network.named_parameters())
        for i, name in enumerate(self.names):
            self.register_buffer(f'ref_{i}', torch.zeros_like(parameters[name], dtype=torch.float32))
        self.register_buffer('ready', torch.tensor(False))

    # จับค่าน้ำหนักอ้างอิงหลังโหลด source; ไม่รวม classifier head
    def capture(self, network):
        if bool(self.ready):
            raise RuntimeError('L2-SP source reference already captured')
        parameters = dict(network.named_parameters())
        with torch.no_grad():
            for i, name in enumerate(self.names):
                getattr(self, f'ref_{i}').copy_(parameters[name].detach().float())
            self.ready.fill_(True)

    # คืนผลรวม squared distance; DLModel คูณ lambda/2 ก่อนบวก training loss
    def forward(self, network):
        if not bool(self.ready):
            raise RuntimeError('L2-SP reference missing: initialize from a strict source checkpoint first')
        parameters = dict(network.named_parameters())
        return sum((parameters[name].float() - getattr(self, f'ref_{i}')).square().sum()
                   for i, name in enumerate(self.names))
