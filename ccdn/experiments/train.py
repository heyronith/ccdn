from __future__ import annotations
import argparse, copy, datetime as dt, subprocess, time, uuid
from pathlib import Path
import torch
import torch.nn.functional as F
from ccdn.models.dense_mlp import DenseMLP
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.baselines import StaticDense,StaticSparse,SelectiveReset,ContinualBackprop,RigL
from ccdn.algorithms import CCDN0A
from ccdn.streams.permuted_mnist import PermutedMNISTStream,permute_batch
from ccdn.metrics import diagnostics,account,normalized_adaptation_auc,summarize_adaptation_auc
from ccdn.utils.config import load_config,save_config
from ccdn.utils.reproducibility import seed_everything
from ccdn.utils.logging import write_metrics,write_summary
from ccdn.utils.checkpointing import save_checkpoint

ALGOS={"static_dense":StaticDense,"static_sparse":StaticSparse,"selective_reset":SelectiveReset,"continual_backprop":ContinualBackprop,"continual_backprop_sparse":ContinualBackprop,"rigl":RigL,"ccdn_0a":CCDN0A}
SPARSE_MODELS={"static_sparse","selective_reset","rigl","continual_backprop_sparse","ccdn_0a"}
DEFAULT_COMPARISON_GROUPS={
    "static_dense":"contextual_dense_reference",
    "continual_backprop":"contextual_dense_reference",
    "static_sparse":"primary_sparse_resource_matched",
    "selective_reset":"primary_sparse_resource_matched",
    "rigl":"primary_sparse_resource_matched",
    "continual_backprop_sparse":"primary_sparse_resource_matched",
    "ccdn_0a":"primary_sparse_resource_matched",
}
def _git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: return None
def _device(value): return torch.device("cuda" if value=="auto" and torch.cuda.is_available() else ("cpu" if value=="auto" else value))
def build(config,device):
    mc=config["model"]; typ=mc["type"]; dims=mc.get("hidden_sizes",[256,256]); seed=config["experiment"]["seed"]
    if typ in SPARSE_MODELS:
        model=SparseMLP(hidden_sizes=dims,density=mc.get("density",0.2),seed=seed)
    else: model=DenseMLP(hidden_sizes=dims)
    model.to(device)
    oc=config.get("optimizer",{}); optimizer=torch.optim.SGD(model.parameters(),lr=float(oc.get("lr",0.01)),momentum=float(oc.get("momentum",0.0)))
    opts=config.get("algorithm",{})
    alg=ALGOS[typ](model,**opts) if opts else ALGOS[typ](model)
    return model,alg,optimizer

def create_stream(stream_cfg,seed):
    """Create the configured stream without silently substituting scientific data."""
    common=dict(root=stream_cfg.get("dataset_root","./data"),seed=seed,
                permutations=stream_cfg.get("permutations",20),
                batches_per_permutation=stream_cfg.get("batches_per_permutation",20),
                batch_size=stream_cfg.get("batch_size",128))
    if stream_cfg.get("synthetic",False):
        return PermutedMNISTStream(**common,synthetic=True),False,None
    if not stream_cfg.get("allow_synthetic_fallback",False):
        return PermutedMNISTStream(**common,synthetic=False,download=True),False,None
    try:
        return PermutedMNISTStream(**common,synthetic=False,download=True),False,None
    except Exception as error:
        reason=f"{type(error).__name__}: {error}"
        return PermutedMNISTStream(**common,synthetic=True),True,reason

def evaluate(model,loader,perm,device,max_batches=2):
    model.eval(); total=correct=0; loss_sum=0.0
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            if i>=max_batches: break
            x=permute_batch(x.to(device),perm.to(device)); y=y.to(device)
            logits=model(x); loss=F.cross_entropy(logits,y,reduction="sum"); loss_sum+=float(loss); correct+=int((logits.argmax(1)==y).sum()); total+=y.numel()
    model.train(); return loss_sum/max(total,1),correct/max(total,1)

def run(config, config_source=None):
    started=time.perf_counter(); exp=config["experiment"]; stream_cfg=config["stream"]; model_cfg=config["model"]
    seed=int(exp.get("seed",1)); seed_everything(seed,exp.get("deterministic",True)); device=_device(exp.get("device","auto"))
    requested_synthetic=stream_cfg.get("synthetic",False)
    stream,fallback,fallback_reason=create_stream(stream_cfg,seed)
    model,algo,optimizer=build(config,device); rows=[]; aucs=[]; samples=0; last_loss=0.
    eval_every=max(1,int(config.get("evaluation",{}).get("eval_every_steps",10))); max_eval=int(config.get("evaluation",{}).get("eval_batches",2))
    for permutation_index,batches,test_loader,perm in stream.segments():
        adaptation=[]; segment_start=algo.global_step
        # Boundary-aware evaluation stays in the runner; only (x, y) reaches the learner.
        el,ea=evaluate(model,test_loader,perm,device,max_eval); adaptation.append((0,ea))
        rows.append({"global_step":algo.global_step,"permutation_index":permutation_index,"samples_seen":samples,"train_loss":last_loss,"eval_loss":el,"eval_accuracy":ea,**account(model,algo,optimizer),**diagnostics(model),**algo.metrics()})
        for x,y in batches:
            x=x.to(device); y=y.to(device); logits=model(x); loss=F.cross_entropy(logits,y); last_loss=float(loss.detach())
            algo.before_backward(loss); loss.backward(); algo.after_backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            algo.after_optimizer_step(optimizer); samples+=y.numel()
            if algo.global_step%eval_every==0:
                el,ea=evaluate(model,test_loader,perm,device,max_eval); adaptation.append((algo.global_step-segment_start,ea))
                row={"global_step":algo.global_step,"permutation_index":permutation_index,"samples_seen":samples,"train_loss":last_loss,"eval_loss":el,"eval_accuracy":ea,**account(model,algo,optimizer),**diagnostics(model),**algo.metrics()}; rows.append(row)
        if not adaptation or adaptation[-1][0] != algo.global_step-segment_start:
            el,ea=evaluate(model,test_loader,perm,device,max_eval); adaptation.append((algo.global_step-segment_start,ea))
            rows.append({"global_step":algo.global_step,"permutation_index":permutation_index,"samples_seen":samples,"train_loss":last_loss,"eval_loss":el,"eval_accuracy":ea,**account(model,algo,optimizer),**diagnostics(model),**algo.metrics()})
        aucs.append({"permutation_index":permutation_index,"adaptation_auc":normalized_adaptation_auc(adaptation),"end_accuracy":adaptation[-1][1]})
    runtime=time.perf_counter()-started
    typ=model_cfg["type"]; outroot=Path(config.get("output",{}).get("root","results")); label=exp.get("name", "pmnist")
    run_dir=outroot/label/typ/f"seed_{seed}"/(dt.datetime.now().strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]); run_dir.mkdir(parents=True,exist_ok=False)
    resolved=copy.deepcopy(config); resolved["runtime_seconds"]=runtime
    save_config(resolved,run_dir/"config.yaml"); write_metrics(rows,run_dir/"metrics.csv"); write_metrics(aucs,run_dir/"adaptation_auc.csv")
    final_resource=account(model,algo,optimizer)
    primary={**summarize_adaptation_auc(aucs),
             "mean_end_accuracy":sum(a["end_accuracy"] for a in aucs)/max(len(aucs),1),
             "last_permutation_end_accuracy":aucs[-1]["end_accuracy"] if aucs else None}
    summary={"timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),"git_commit":_git_commit(),"config":resolved,"seed":seed,"device":str(device),"model_type":typ,"comparison_group":exp.get("comparison_group",DEFAULT_COMPARISON_GROUPS.get(typ,"unassigned")),"number_of_permutations":stream_cfg.get("permutations",20),"samples_per_permutation":stream_cfg.get("batches_per_permutation",20)*stream_cfg.get("batch_size",128),"updates_per_permutation":stream_cfg.get("batches_per_permutation",20),"active_parameters":final_resource["logical_active_parameters"],"total_parameters":final_resource["total_possible_parameters"],"auxiliary_state_estimate_bytes":final_resource["auxiliary_algorithm_state_memory_bytes"],"optimizer_state_estimate_bytes":final_resource["optimizer_state_memory_bytes"],"runtime_seconds":runtime,"primary_metrics":primary,"resources":final_resource,"data_source":"synthetic" if requested_synthetic or fallback else "mnist","synthetic_fallback":fallback}
    if fallback: summary["fallback_reason"]=fallback_reason
    write_summary(summary,run_dir/"summary.json"); save_checkpoint(run_dir/"checkpoint.pt",model,algo,optimizer,algo.global_step,resolved)
    print(run_dir)
    return run_dir

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); run(load_config(a.config),a.config)
if __name__=="__main__": main()
