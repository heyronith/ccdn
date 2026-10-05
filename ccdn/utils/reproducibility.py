import os, random
import numpy as np
import torch

def seed_everything(seed: int, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.use_deterministic_algorithms(True,warn_only=True)
        if hasattr(torch.backends,"cudnn"):
            torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def capture_rng_state():
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"]=torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])
