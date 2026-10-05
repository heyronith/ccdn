import argparse,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument("path"); a=p.parse_args(); files=list(Path(a.path).rglob("summary.json")); rows=[]
    for f in files:
        d=json.loads(f.read_text()); rows.append({"path":str(f.parent),"model":d["model_type"],"seed":d["seed"],**d["primary_metrics"]})
    out=Path(a.path)/"aggregate_summary.csv"; pd.DataFrame(rows).to_csv(out,index=False); print(out)
if __name__=="__main__": main()
