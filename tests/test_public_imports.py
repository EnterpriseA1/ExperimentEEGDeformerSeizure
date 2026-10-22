"""Public model imports must respect the caller's selected GPU devices."""
import os
import subprocess
import sys
from pathlib import Path


def test_model_import_preserves_device_selection():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',CUDA_DEVICE_ORDER='FASTEST_FIRST')
    result=subprocess.run([sys.executable,'-c',
        "import os; import models.model_handler; "
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='7'; "
        "assert os.environ['CUDA_DEVICE_ORDER']=='FASTEST_FIRST'"],
        cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
