from pathlib import Path
import yaml

def load_config(path):
    with open(path,encoding="utf8") as f: return yaml.safe_load(f)
def save_config(config,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf8") as f: yaml.safe_dump(config,f,sort_keys=False)
