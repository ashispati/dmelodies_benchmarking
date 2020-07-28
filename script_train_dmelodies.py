import os
import json
import torch
from dmelodies_torch_dataloader import DMelodiesTorchDataset
from model.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from model.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer

m = 'beta-VAE'
# m = 'annealed-VAE'

seed_list = [0, 1, 2]
model_dict = {
    'beta-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2, 1.0, 4.0]
    },
    'annealed-VAE': {
        'capacity_list': [25.0, 50.0, 75.0],
        'beta_list': [1.0]
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
            trainer = DMelodiesVAETrainer(
                dataset,
                vae_model,
                model_type=m,
                beta=b,
                capacity=c,
                lr=1e-4,
                rand=seed
            )
            if not os.path.exists(vae_model.filepath):
                trainer.train_model(batch_size=batch_szie, num_epochs=num_epochs, log=False)
            else:
                print('Model exists. Running evaluation.')
            trainer.load_model()
            metrics = trainer.compute_eval_metrics()
            print(json.dumps(metrics, indent=2))
