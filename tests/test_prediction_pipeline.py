"""Public inference -> saved cache -> metric report with a real tiny checkpoint."""
import json
import pickle
import sys
from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from evaluation import predict, compute_metrics
from models.model_handler import DLModel
from utils import EEGDataset


def test_checkpoint_to_window_report(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(2024)
    config = Namespace(model='EEGNet', num_chan=16, num_time=800, num_class=2,
                       dropout=0., sampling_rate=200, norm_mode='legacy_segment_f16')
    model = DLModel(config)
    model.optimal_threshold.fill_(.37)
    ckpt = tmp_path/'run_1/checkpoints/best.ckpt'
    ckpt.parent.mkdir(parents=True)
    torch.save({'state_dict':model.state_dict(), 'hyper_parameters':{'config':config},
                'pytorch-lightning_version':pl.__version__}, ckpt)
    data_dir=tmp_path/'data/data_eeg_TUSZ_seizure_eval'
    data_dir.mkdir(parents=True)
    x=np.random.default_rng(2024).normal(0,1e-5,(4,16,800)).astype(np.float32)
    y=np.array([0,1,0,1])
    with (data_dir/'sub0.pkl').open('wb') as f:pickle.dump({'data':x,'label':y},f)
    channels=[f'ch{i}' for i in range(16)]
    with (data_dir/'dataset_info.pkl').open('wb') as f:pickle.dump({'BL':channels},f)
    np.savez(tmp_path/'boundaries.npz',verified=True,subject_ids=[0],boundaries=[[0,2]])
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def loader(data_list,label_list,idx,batch_size,shuffle,norm_mode):
        return DataLoader(EEGDataset(data_list,label_list,idx_channels=idx,norm_mode=norm_mode),
                          batch_size=batch_size,shuffle=shuffle,num_workers=0)
    monkeypatch.setattr(predict,'prepare_data_for_training',loader)
    monkeypatch.setattr(sys,'argv',['predict','--checkpoints',f'EEGNet_seed2024={ckpt}',
                                  '--load-path',str(tmp_path/'data'),'--boundaries','boundaries.npz',
                                  '--out','predictions.npz'])
    predict.main()
    cache=np.load('predictions.npz')
    assert cache['probs'].shape==(1,4)
    np.testing.assert_array_equal(cache['labels'],y)
    np.testing.assert_array_equal(cache['seam'],[True,False,True,False])
    assert abs(float(cache['thresholds'][0])-.37)<1e-6
    assert np.isfinite(cache['probs']).all()
    monkeypatch.setattr(sys,'argv',['metrics','predictions.npz','--out-prefix','results/report'])
    compute_metrics.main()
    report=json.loads((tmp_path/'results/report_summary.json').read_text())
    g=report['runs'][0]['global']
    assert g['n_windows']==4
    assert g['tp']+g['tn']+g['fp']+g['fn']==4
    assert abs(g['threshold_from_dev']-.37)<1e-6
