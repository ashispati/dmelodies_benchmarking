import os
import json
import torch
from dmelodies_torch_dataloader import DMelodiesTorchDataset
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.dmelodiesvae.dmelodies_cnnvae import DMelodiesCNNVAE
from src.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer
from src.dmelodiesvae.dmelodies_cnnvae_trainer import DMelodiesCNNVAETrainer
from src.dmelodiesvae.interp_vae import InterpVAE
from src.dmelodiesvae.interp_vae_trainer import InterpVAETrainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    default='beta-VAE',
    choices=['beta-VAE', 'annealed-VAE', 'ar-VAE', 'interp-VAE']
)
parser.add_argument("--net_type", type=str, default='rnn', choices=['rnn', 'cnn'])
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--delta", type=float, default=10.0)
parser.add_argument("--interp_num_dims", type=int, default=None)
parser.add_argument("--no_log", action='store_false')

args = parser.parse_args()

# Select the Type of VAE-model and the network architecture
m = args.model_type
net_type = args.net_type

# Specify training params
seed_list = [0, 1, 2]
model_dict = {
    'beta-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2, 1.0, 4.0]
    },
    'annealed-VAE': {
        'capacity_list': [25.0, 50.0, 75.0],
        'beta_list': [1.0]
    },
    'ar-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma': args.gamma,
        'delta': args.delta,
    },
    'interp-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma': args.gamma,
        'num_dims': args.interp_num_dims
    }
}
num_epochs = 100
batch_size = 512

# Specify the network and trainer classes
if m == 'interp-VAE':
    model = InterpVAE
    trainer = InterpVAETrainer
else:
    if net_type == 'cnn':
        model = DMelodiesCNNVAE
        trainer = DMelodiesCNNVAETrainer
    else:
        model = DMelodiesVAE
        trainer = DMelodiesVAETrainer


c_list = model_dict[m]['capacity_list']
b_list = model_dict[m]['beta_list']
for seed in seed_list:
    for c in c_list:
        for b in b_list:
            dataset = DMelodiesTorchDataset(seed=seed)
            if m == 'interp-VAE':
                vae_model = model(dataset, vae_type=net_type, num_dims=model_dict[m]['num_dims'])
            else:
                vae_model = model(dataset)
            if torch.cuda.is_available():
                vae_model.cuda()
            trainer_args = {
                'model_type': m,
                'beta': b,
                'capacity': c,
                'lr': 1e-4,
                'rand': seed
            }
            if m == 'ar-VAE':
                trainer_args.update({'gamma': model_dict[m]['gamma']})
                trainer_args.update({'delta': model_dict[m]['delta']})
            elif m == 'interp-VAE':
                trainer_args.update({'gamma': model_dict[m]['gamma']})
            vae_trainer = trainer(
                dataset,
                vae_model,
                **trainer_args
            )
            if not os.path.exists(vae_model.filepath):
                vae_trainer.train_model(batch_size=batch_size, num_epochs=num_epochs, log=args.no_log)
            else:
                print('Model exists. Running evaluation.')
            vae_trainer.load_model()
            metrics = vae_trainer.compute_eval_metrics()
            print(json.dumps(metrics, indent=2))
