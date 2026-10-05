import math,torch
from ccdn.metrics.continual import normalized_adaptation_auc,early_late_summary
from ccdn.metrics.plasticity import diagnostics
from ccdn.models.dense_mlp import DenseMLP

def test_auc_and_early_late():
    assert abs(normalized_adaptation_auc([(0,0),(1,1)])-.5)<1e-8
    rows=[{"eval_accuracy":float(i)} for i in range(8)]; s=early_late_summary(rows)
    assert s["early_25pct_accuracy"]==.5 and s["late_25pct_accuracy"]==6.5

def test_diagnostics_finite():
    m=DenseMLP(input_size=4,hidden_sizes=(3,),output_size=2); m(torch.ones(2,4)); d=diagnostics(m)
    assert all(math.isfinite(v) for v in d.values())

def test_lifetime_auc_summary_uses_permutation_indices(tmp_path):
    import pandas as pd
    from ccdn.metrics.continual import summarize_adaptation_auc
    rows=[{'permutation_index':i,'adaptation_auc':i/100,'end_accuracy':i/100} for i in range(100)]
    path=tmp_path/'adaptation_auc.csv'; pd.DataFrame(rows).to_csv(path,index=False)
    summary=summarize_adaptation_auc(path)
    assert abs(summary['mean_adaptation_auc']-.495)<1e-9
    assert abs(summary['first_25pct_mean_auc']-.12)<1e-9
    assert abs(summary['last_25pct_mean_auc']-.87)<1e-9
    assert abs(summary['late_minus_early_auc']-.75)<1e-9
    assert abs(summary['auc_lifetime_slope']-.01)<1e-9
