import wandb
import json
from train import train_args_parser
import numpy as np
from tqdm import tqdm
api = wandb.Api()

args = train_args_parser()

# get entity and project
with open(args.wandb_config) as f:
    wandb_config = json.load(f)

# login
wandb.login(key=wandb_config['key'])

# get sweep id
sweep_id = args.sweep_id

# get results
runs = wandb.Api().runs(path=f"{wandb_config['entity']}/{wandb_config['project']}", filters={"sweep":f"{sweep_id}"})
summary_list = []
for run in tqdm(runs):
    # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)


import pandas as pd
summary_df = pd.DataFrame.from_records(summary_list)
all_df = pd.concat([summary_df], axis=1)

all_df.to_csv("../outputs/sweep_outputs/{}.csv".format(args.table_name))