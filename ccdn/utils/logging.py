import json
from pathlib import Path
import pandas as pd

def write_metrics(rows,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(path,index=False)
def write_summary(data,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf8") as f: json.dump(data,f,indent=2,allow_nan=False,default=str)
