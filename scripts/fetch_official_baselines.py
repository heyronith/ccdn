#!/usr/bin/env python3
"""Fetch and verify the official source repositories at their pinned SHAs."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.external'/'official_baselines'
SOURCES={
    'loss-of-plasticity':('https://github.com/shibhansh/loss-of-plasticity.git','a6b79580d85f3025bdb601566d3627c5f489f13b'),
    'rigl':('https://github.com/google-research/rigl.git','d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9'),
}

def git(*args,cwd=None):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()

def fetch_one(name,url,expected):
    path=DEST/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        git('clone',url,str(path))
    elif not (path/'.git').exists():
        raise RuntimeError(f'{path} exists but is not a Git checkout')
    else:
        remote=git('remote','get-url','origin',cwd=path).rstrip('/')
        if remote.endswith('.git'): remote=remote[:-4]
        wanted=url.rstrip('/')
        if wanted.endswith('.git'): wanted=wanted[:-4]
        if remote!=wanted:
            raise RuntimeError(f'{path} origin mismatch: {remote} != {wanted}')
        git('fetch','--tags','origin',cwd=path)
    git('checkout','--detach',expected,cwd=path)
    actual=git('rev-parse','HEAD',cwd=path)
    print(f'{name} official source:\nexpected: {expected}\nactual:   {actual}')
    if actual!=expected:
        raise RuntimeError(f'{name} commit mismatch: {actual} != {expected}')
    print('VERIFIED')
    return actual

def main():
    for name,(url,sha) in SOURCES.items(): fetch_one(name,url,sha)

if __name__=='__main__': main()
