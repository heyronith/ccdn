import numpy as np

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
