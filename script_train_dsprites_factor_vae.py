import os
import torch
from src.dspritesvae.dsprites_torch_dataset import DspritesDataset
from src.dspritesvae.dsprites_factorvae import DspritesFactorVAE
from src.dspritesvae.image_factorvae_trainer import ImageFactorVAETrainer

gamma_list = [1, 10, 50]
capacity_list = [50, 1]
beta_list = [1]
seed_list = [0, 1, 2]
num_epochs = 100
batch_size = 1024

for seed in seed_list:
    for c in capacity_list:
        for b in beta_list:
            for g in gamma_list:
                dataset = DspritesDataset(seed=seed)
                vae_model = DspritesFactorVAE()
                if torch.cuda.is_available():
                    vae_model.cuda()
                trainer = ImageFactorVAETrainer(
                    dataset,
                    vae_model,
                    beta=b,
                    capacity=c,
                    gamma=g,
                    lr=1e-4,
                    rand=seed
                )
                if not os.path.exists(vae_model.filepath):
                    trainer.train_model(batch_size=batch_size, num_epochs=num_epochs, log=True)
                else:
                    print('Model exists. Running evaluation.')
                trainer.load_model()
                metrics = trainer.compute_eval_metrics()
                print(metrics)
