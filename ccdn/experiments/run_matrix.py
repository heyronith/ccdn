import argparse,copy
from ccdn.experiments.train import run
from ccdn.utils.config import load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--suite",required=True); a=p.parse_args(); suite=load_config(a.suite)
    base=suite.get("base",{}); results=[]
    for model in suite["models"]:
        for seed in suite["seeds"]:
            cfg=copy.deepcopy(base); cfg.setdefault("experiment",{})["seed"]=seed
            cfg.setdefault("model",{})["type"]=model
            if model in suite.get("comparison_groups",{}):
                cfg["experiment"]["comparison_group"]=suite["comparison_groups"][model]
            # Keep suite output unique by model/seed while retaining shared schema.
            cfg.setdefault("experiment",{}).setdefault("name",suite.get("name","smoke"))
            if model in suite.get("model_overrides",{}): cfg["model"].update(suite["model_overrides"][model])
            if model in suite.get("algorithm_overrides",{}): cfg["algorithm"]=suite["algorithm_overrides"][model]
            results.append(run(cfg))
    print(f"Completed {len(results)} runs")
if __name__=="__main__": main()
