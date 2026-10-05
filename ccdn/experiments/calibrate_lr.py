"""Run the preregistered shared static-model learning-rate calibration."""
import argparse, copy, json
from pathlib import Path
import pandas as pd
from ccdn.experiments.train import run
from ccdn.utils.config import load_config, save_config

def calibrate(config_path):
    calibration=load_config(config_path); records=[]
    for lr in calibration['learning_rates']:
        for model_type in calibration['models']:
            cfg=copy.deepcopy(calibration['base'])
            cfg['experiment']['name']=calibration['name']
            cfg['experiment']['seed']=int(calibration['seed'])
            cfg['model']['type']=model_type
            cfg['optimizer']['lr']=float(lr)
            run_dir=run(cfg)
            summary=json.loads((run_dir/'summary.json').read_text())
            if summary['data_source']!='mnist' or summary['synthetic_fallback']:
                raise RuntimeError(f"Calibration must use MNIST; got {summary['data_source']} at {run_dir}")
            records.append({
                'learning_rate':float(lr),'model':model_type,'seed':int(calibration['seed']),
                'comparison_group':'contextual_dense_reference' if model_type=='static_dense' else 'primary_sparse_resource_matched',
                'permutations':5,'batches_per_permutation':20,
                'mean_end_accuracy':summary['primary_metrics']['mean_end_accuracy'],
                'run_directory':str(run_dir),
            })
    frame=pd.DataFrame(records)
    means=frame.groupby('learning_rate',as_index=False)['mean_end_accuracy'].mean().rename(columns={'mean_end_accuracy':'selection_mean_static_dense_sparse'})
    # Treat differences below 1e-12 as tied and prefer the smaller shared rate.
    best=means.selection_mean_static_dense_sparse.max()
    chosen=float(means.loc[means.selection_mean_static_dense_sparse >= best-1e-12,'learning_rate'].min())
    frame=frame.merge(means,on='learning_rate'); frame['chosen']=frame.learning_rate==chosen
    out=Path('review_artifacts/cycle_2'); out.mkdir(parents=True,exist_ok=True)
    frame.sort_values(['learning_rate','model']).to_csv(out/'lr_calibration.csv',index=False)
    (out/'lr_calibration_selection.json').write_text(json.dumps({
        'chosen_shared_learning_rate':chosen,
        'selection_rule':'highest mean end-of-permutation accuracy averaged across static_dense and static_sparse over five permutations; ties within 1e-12 choose smaller LR',
        'per_learning_rate_means':means.sort_values('learning_rate').to_dict(orient='records'),
        'data_source':'mnist','seed':int(calibration['seed']),
    },indent=2)+'\n')
    for path in ('configs/suites/cycle2_preflight.yaml','configs/suites/cycle2_pilot.yaml'):
        cfg=load_config(path); cfg['base']['optimizer']['lr']=chosen; save_config(cfg,path)
    print(f"Chosen shared learning rate: {chosen}")
    print(out/'lr_calibration.csv')
    return chosen

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/cycle2/calibration.yaml'); args=parser.parse_args(); calibrate(args.config)
if __name__=='__main__': main()
