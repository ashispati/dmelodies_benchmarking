import os
import json
import torch
from dmelodies_torch_dataloader import DMelodiesTorchDataset
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='beta-VAE', choices=['beta-VAE', 'annealed-VAE', 'ar-VAE'])
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--delta", type=float, default=10.0)

args = parser.parse_args()

# Select the Type of VAE-model
m = args.model_type

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
        'beta_list': [0.001],
        'gamma': args.gamma,
        'delta': args.delta,
    }
}
num_epochs = 100
batch_szie = 512

c_list = model_dict[m]['capacity_list']
b_list = model_dict[m]['beta_list']
for seed in seed_list:
    for c in c_list:
        for b in b_list:
            dataset = DMelodiesTorchDataset(seed=seed)
            vae_model = DMelodiesVAE(dataset)
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
            trainer = DMelodiesVAETrainer(
                dataset,
                vae_model,
                **trainer_args
            )
            if not os.path.exists(vae_model.filepath):
                trainer.train_model(batch_size=batch_szie, num_epochs=num_epochs, log=True)
            else:
                print('Model exists. Running evaluation.')
            trainer.load_model()
            metrics = trainer.compute_eval_metrics()
            print(json.dumps(metrics, indent=2))
