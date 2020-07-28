import os
import json
import torch
from src.dspritesvae.dsprites_torch_dataset import DspritesDataset
from src.dspritesvae.dsprites_vae import DspritesVAE
from src.dspritesvae.image_vae_trainer import ImageVAETrainer


# Select the Type of VAE-model
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
batch_size = 512

c_list = model_dict[m]['capacity_list']
b_list = model_dict[m]['beta_list']
for seed in seed_list:
    for c in c_list:
        for b in b_list:
            dataset = DspritesDataset(seed=seed)
            vae_model = DspritesVAE()
            if torch.cuda.is_available():
                vae_model.cuda()
            trainer = ImageVAETrainer(
                dataset,
                vae_model,
                model_type=m,
                beta=b,
                capacity=c,
                lr=1e-4,
                rand=seed
            )
            if not os.path.exists(vae_model.filepath):
                trainer.train_model(batch_size=batch_size, num_epochs=num_epochs, log=False)
            else:
                print('Model exists. Running evaluation.')
            trainer.load_model()
            metrics = trainer.compute_eval_metrics()
            print(json.dumps(metrics, indent=2))
