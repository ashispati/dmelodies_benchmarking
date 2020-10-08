import os
from dmelodies_torch_dataloader import DMelodiesTorchDataset
from src.dmelodiesvae.dmelodies_cnnvae import DMelodiesCNNVAE
from src.dmelodiesvae.dmelodies_cnnvae_trainer import DMelodiesCNNVAETrainer

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

c_list = model_dict[m]['capacity_list']
b_list = model_dict[m]['beta_list']
for c in c_list:
    for b in b_list:
        for seed in seed_list:
            dataset = DMelodiesTorchDataset(seed=seed)
            vae_model = DMelodiesCNNVAE(dataset).cuda()
            trainer = DMelodiesCNNVAETrainer(
                dataset,
                vae_model,
                model_type=m,
                beta=b,
                capacity=c,
                lr=1e-4,
                rand=seed
            )
            if not os.path.exists(vae_model.filepath):
                try:
                    trainer.train_model(batch_size=512, num_epochs=num_epochs, log=True)
                except Exception as e:
                    with open('errors.txt', 'a') as f:
                        f.write('{} failed after {} epochs'.format(vae_model.filepath, trainer.cur_epoch_num))
                    continue
            else:
                print('Model exists. Running evaluation.')
            trainer.load_model()
            metrics = trainer.compute_eval_metrics()
            print(metrics)
