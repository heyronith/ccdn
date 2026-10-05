"""Build compact lifetime and resource tables from completed Cycle 2 pilot runs."""
import argparse, json
from pathlib import Path
import pandas as pd
from ccdn.metrics.continual import summarize_adaptation_auc

SPARSE_PRIMARY=['static_sparse','selective_reset','rigl','continual_backprop_sparse','ccdn_0a']
DENSE_CONTEXT=['static_dense','continual_backprop']

def _summarize_runs(root):
    records=[]; curves=[]
    for summary_path in sorted(Path(root).rglob('summary.json')):
        summary=json.loads(summary_path.read_text())
        auc_path=summary_path.parent/'adaptation_auc.csv'
        auc=pd.read_csv(auc_path)
        eval_rows=pd.read_csv(summary_path.parent/'metrics.csv')
        if summary['data_source']!='mnist' or summary['synthetic_fallback']:
            raise RuntimeError(f"Pilot analysis refuses non-MNIST run: {summary_path}")
        if len(auc)!=summary['number_of_permutations']:
            raise RuntimeError(f"Incomplete per-permutation AUC in {auc_path}")
        lifetime=summarize_adaptation_auc(auc_path)
        model=summary['model_type']; seed=int(summary['seed']); group=summary['comparison_group']
        record={'model':model,'seed':seed,'comparison_group':group,'data_source':summary['data_source'],**lifetime,
                'mean_end_accuracy':float(auc.end_accuracy.mean()),'last_permutation_end_accuracy':float(auc.end_accuracy.iloc[-1]),
                'runtime_seconds':summary['runtime_seconds'],**summary['resources']}
        final_row=eval_rows.sort_values('global_step').iloc[-1]
        for key in ('rewire_event_count','rewire_count','total_edges_pruned','total_edges_grown',
                    'cumulative_edge_turnover','reset_count','replacement_count',
                    'mean_active_utility','std_active_utility'):
            if key in eval_rows.columns:
                record[key]=final_row[key]
        records.append(record)
        for row in auc.to_dict(orient='records'):
            curves.append({'model':model,'seed':seed,'comparison_group':group,**row})
    return pd.DataFrame(records),pd.DataFrame(curves)

def make_paired_differences(summaries):
    rows=[]
    metrics=('mean_adaptation_auc','first_25pct_mean_auc','last_25pct_mean_auc',
             'late_minus_early_auc','auc_lifetime_slope','mean_end_accuracy')
    dense=summaries[summaries.model=='ccdn_0a'].set_index('seed')
    for comparator in SPARSE_PRIMARY[:-1]:
        other=summaries[summaries.model==comparator].set_index('seed')
        for seed in sorted(set(dense.index)&set(other.index)):
            for metric in metrics:
                rows.append({'comparison_group':'primary_sparse_resource_matched','metric':metric,'seed':seed,'ccdn_0a':float(dense.loc[seed,metric]),
                             'comparator':comparator,'comparator_value':float(other.loc[seed,metric]),
                             'paired_difference_ccdn_minus_comparator':float(dense.loc[seed,metric]-other.loc[seed,metric]),
                             'interpretation':'raw paired pilot value; no significance claim'})
    return pd.DataFrame(rows)

def analyze(pilot_root,preflight_root,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    pilot,curves=_summarize_runs(pilot_root); preflight,_=_summarize_runs(preflight_root)
    expected_models=set(DENSE_CONTEXT+SPARSE_PRIMARY); expected_seeds={101,202,303}
    if set(pilot.model)!=expected_models or set(pilot.seed)!=expected_seeds or len(pilot)!=21:
        raise RuntimeError(f"Pilot matrix incomplete: {len(pilot)} runs")
    if set(preflight.model)!=expected_models or set(preflight.seed)!={11} or len(preflight)!=7:
        raise RuntimeError(f"Preflight matrix incomplete: {len(preflight)} runs")
    pilot.sort_values(['model','seed']).to_csv(out/'pilot_summaries.csv',index=False)
    preflight.sort_values(['model','seed']).to_csv(out/'preflight_summaries.csv',index=False)
    curves.sort_values(['model','seed','permutation_index']).to_csv(out/'permutation_auc.csv',index=False)
    make_paired_differences(pilot).to_csv(out/'paired_sparse_differences.csv',index=False)
    resources=pilot.groupby(['model','comparison_group'],as_index=False).agg({
        'logical_active_parameters':'mean','actual_tensor_parameters':'mean','model_tensor_memory_bytes':'mean',
        'auxiliary_algorithm_state_memory_bytes':'mean','optimizer_state_memory_bytes':'mean',
        'estimated_total_training_state_memory_bytes':'mean','estimated_dense_macs_per_example':'mean',
        'runtime_seconds':'mean'})
    resources.to_csv(out/'resource_comparison.csv',index=False)
    manifest={'cycle':2,'pilot_label':'Pilot experiment — not definitive evidence.',
              'data_source':'MNIST','synthetic_fallback':False,'pilot_runs':len(pilot),
              'preflight_runs':len(preflight),'models':sorted(expected_models),'pilot_seeds':[101,202,303],
              'permutations':100,'batches_per_permutation':50,'batch_size':128,
              'learning_rate':float(pd.read_csv(out/'lr_calibration.csv').loc[lambda d:d.chosen,'learning_rate'].iloc[0]),
              'comparison_groups':{'primary_sparse_resource_matched':SPARSE_PRIMARY,'contextual_dense_reference':DENSE_CONTEXT},
              'paired_differences':'Raw per-seed CCDN-0A minus comparator values; no p-values or significance claims.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"Analyzed {len(preflight)} preflight and {len(pilot)} pilot runs into {out}")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pilot-root',default='results/cycle2_pilot'); parser.add_argument('--preflight-root',default='results/cycle2_preflight'); parser.add_argument('--out',default='review_artifacts/cycle_2'); args=parser.parse_args(); analyze(args.pilot_root,args.preflight_root,args.out)
if __name__=='__main__': main()
