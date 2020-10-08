import os
from dmelodies_torch_dataloader import DMelodiesTorchDataset
from src.dmelodiesvae.factor_vae import FactorVAE
from src.dmelodiesvae.factor_vae_trainer import FactorVAETrainer

gamma_list = [1, 10, 50]
capacity_list = [50]
beta_list = [1, 0.2]
seed_list = [0, 1, 2]
vae_type = ['CNN', 'RNN']
num_epochs = 100
batch_size = 512

for g in gamma_list:
    for c in capacity_list:
        for b in beta_list:
            for seed in seed_list:
                for v_type in vae_type:
                    dataset = DMelodiesTorchDataset(seed=seed)
                    vae_model = FactorVAE(dataset, vae_type=v_type).cuda()
                    trainer = FactorVAETrainer(
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
