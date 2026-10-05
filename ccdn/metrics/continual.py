import numpy as np
from pathlib import Path

def normalized_adaptation_auc(points):
    """Trapezoidal accuracy AUC over normalized within-permutation time."""
    if len(points)<2: return float(points[0][1]) if points else float("nan")
    x=np.asarray([p[0] for p in points],dtype=float); y=np.asarray([p[1] for p in points],dtype=float)
    if x[-1]==x[0]: return float(y.mean())
    return float(np.trapezoid(y,(x-x[0])/(x[-1]-x[0])))

def early_late_summary(rows):
    n=max(1,len(rows)//4)
    vals=[float(r["eval_accuracy"]) for r in rows]
    return {"early_25pct_accuracy":float(np.mean(vals[:n])) if vals else float("nan"),"late_25pct_accuracy":float(np.mean(vals[-n:])) if vals else float("nan")}

def summarize_adaptation_auc(adaptation_auc):
    """Summarize AUC by permutation lifetime, optionally reading a CSV path."""
    if isinstance(adaptation_auc,(str,Path)):
        import pandas as pd
        rows=pd.read_csv(adaptation_auc).to_dict(orient="records")
    else:
        rows=list(adaptation_auc)
    rows=sorted(rows,key=lambda row:int(row["permutation_index"]))
    values=np.asarray([float(row["adaptation_auc"]) for row in rows],dtype=float)
    indices=np.asarray([int(row["permutation_index"]) for row in rows],dtype=float)
    if not len(values):
        return {"mean_adaptation_auc":float("nan"),"first_25pct_mean_auc":float("nan"),"last_25pct_mean_auc":float("nan"),"late_minus_early_auc":float("nan"),"auc_lifetime_slope":float("nan")}
    n=max(1,int(np.ceil(len(values)*0.25)))
    early=float(values[:n].mean()); late=float(values[-n:].mean())
    slope=float(np.polyfit(indices,values,1)[0]) if len(values)>1 else 0.0
    return {"mean_adaptation_auc":float(values.mean()),"first_25pct_mean_auc":early,"last_25pct_mean_auc":late,"late_minus_early_auc":late-early,"auc_lifetime_slope":slope}
