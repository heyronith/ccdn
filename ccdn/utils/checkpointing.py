from pathlib import Path
import torch
from .reproducibility import capture_rng_state,restore_rng_state

def save_checkpoint(path,model,algorithm,optimizer,global_step,config):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"algorithm":algorithm.state_dict(),"optimizer":optimizer.state_dict(),"global_step":global_step,"config":config,"rng":capture_rng_state()},path)
def load_checkpoint(path,model,algorithm,optimizer,map_location="cpu",restore_rng=True):
    state=torch.load(path,map_location=map_location,weights_only=False)
    model.load_state_dict(state["model"]); algorithm.load_state_dict(state["algorithm"]); optimizer.load_state_dict(state["optimizer"])
    if restore_rng and "rng" in state: restore_rng_state(state["rng"])
    return state
