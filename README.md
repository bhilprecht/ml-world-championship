# ML_DHBW_Project
DHBW Semester 6

## Setup
1. Type in Anaconda Prompt: ```jupyter notebook --generate-config```
2. Navigate to folder and append to python script: 
```
import os
from subprocess import check_call

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    d, fname = os.path.split(os_path)
    check_call(['jupyter', 'nbconvert', '--to', 'script', fname], cwd=d)

c.FileContentsManager.post_save_hook = post_save
```