

from train import train_args_parser, train
import wandb
from functools import partial
from datetime import datetime
import yaml
import json
import numpy as np

now = datetime.now()

print("now =", now)

dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

args = train_args_parser()

# get entity and project
with open(args.wandb_config) as f:
    wandb_config = json.load(f)

# login
wandb.login(key=wandb_config['key'])


# set up sweep
sweep_config_file = args.sweep_config
if sweep_config_file is not None:
    with open(sweep_config_file, 'r', encoding='utf-8') as stream:
        sweep_config = yaml.safe_load(stream)
    sweep_config.update({"name": "Sweep started {}".format(dt_string)})

func = partial(train, args=args)

# start the sweep

sweep_id = args.sweep_id
if sweep_id is None:
    sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'], entity=wandb_config['entity'])
else:
    sweep_id = '{}/{}/{}'.format(wandb_config['entity'], wandb_config['project'], sweep_id)
wandb.agent(sweep_id, function=func)