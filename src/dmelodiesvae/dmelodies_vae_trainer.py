import os
import json
import torch
from tqdm import tqdm
import music21
from typing import Tuple

from src.utils.trainer import Trainer
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
from src.utils.evaluation import *

LATENT_ATTRIBUTES = {
    'tonic': 0,
    'octave': 1,
    'mode': 2,
    'rhythm_bar1': 3,
    'rhythm_bar2': 4,
    'arp_chord1': 5,
    'arp_chord2': 6,
    'arp_chord3': 7,
    'arp_chord8': 8
}

LATENT_NORMALIZATION_FACTORS = torch.tensor(
    [11, 2, 2, 27, 27, 1, 1, 1, 1],
    dtype=torch.float32
)


class DMelodiesVAETrainer(Trainer):
    def __init__(
            self,
            dataset,
            model: DMelodiesVAE,
            model_type='beta-VAE',
            lr=1e-4,
            beta=0.001,
            capacity=0.0,
            device=0,
            rand=0,
    ):
        super(DMelodiesVAETrainer, self).__init__(dataset, model, lr)
        self.model_type = model_type
        self.attr_dict = LATENT_ATTRIBUTES
        self.attr_norm_factors = LATENT_NORMALIZATION_FACTORS
        self.reverse_attr_dict = {
            v: k for k, v in self.attr_dict.items()
        }
        self.metrics = {}
        self.beta = beta
        self.capacity = capacity
        # self.capacity = to_cuda_variable(torch.FloatTensor([capacity]))
        self.cur_epoch_num = 0
        self.warm_up_epochs = 10
        self.num_iterations = 100000
        if self.model_type == 'beta-VAE':
            self.exp_rate = np.log(1 + self.beta) / self.num_iterations
            self.start_beta = 0.0
            self.cur_beta = self.start_beta
            self.start_capacity = self.capacity
            self.cur_capacity = self.capacity
        elif self.model_type == 'annealed-VAE':
            self.exp_rate = np.log(1 + self.capacity) / self.num_iterations
            self.start_beta = 0.0
            self.cur_beta = self.start_beta
            self.start_capacity = 0.0
            self.cur_capacity = self.start_capacity
        elif self.model_type == 'ar-VAE':
            self.exp_rate = np.log(1 + self.beta) / self.num_iterations
            self.start_beta = 0.0
            self.cur_beta = self.start_beta
            self.start_capacity = self.capacity
            self.cur_capacity = self.capacity
            self.gamma = 1.0
            self.delta = 10.0
        self.anneal_iterations = 0
        self.device = device
        self.rand_seed = rand
        torch.manual_seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        self.trainer_config = f'_{self.model_type}_b_{self.beta}_c_{self.capacity}_r_{self.rand_seed}_'
        self.model.update_trainer_config(self.trainer_config)

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        if epoch_num > self.warm_up_epochs:
            if self.anneal_iterations < self.num_iterations:
                if self.model_type == 'beta-VAE':
                    self.cur_beta = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
                elif self.model_type == 'annealed-VAE':
                    self.cur_beta = self.beta
                    self.cur_capacity = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
                elif self.model_type == 'ar-VAE':
                    self.cur_beta = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
            self.anneal_iterations += 1

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, latent_tensor = batch
        # convert input to torch Variables
        batch_data = (
            to_cuda_variable_long(score_tensor.squeeze(1), self.device),
            to_cuda_variable_long(latent_tensor.squeeze(1), self.device)
        )
        return batch_data

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param batch_num: int
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        if self.cur_epoch_num != epoch_num:
            flag = True
            self.cur_epoch_num = epoch_num
        else:
            flag = False

        # extract data
        score, latent_attributes = batch

        # perform forward pass of src
        weights, samples, z_dist, prior_dist, z_tilde, z_prior = self.model(
            measure_score_tensor=score,
            measure_metadata_tensor=None,
            train=train
        )

        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(x=score, x_recons=weights)

        # compute KLD loss
        if self.model_type == 'beta-VAE':
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=0.0)
            dist_loss = torch.nn.functional.relu(dist_loss - self.cur_capacity)
        elif self.model_type == 'annealed-VAE':
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=self.cur_capacity)
        elif self.model_type == 'ar-VAE':
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=0.0)
            dist_loss = torch.nn.functional.relu(dist_loss - self.cur_capacity)
        else:
            raise ValueError('Invalid Model Type')

        # add loses
        loss = recons_loss + dist_loss

        # add regularization loss for ar-VAE
        reg_loss = 0.0
        if self.model_type == 'ar-VAE':
            # process latent attributes
            metadata = self.normalize_latent_attributes(latent_attributes)
            # compute regularization loss
            for attr in self.attr_dict.keys():
                dim = self.attr_dict[attr]
                labels = metadata[:, dim]
                reg_loss += self.compute_reg_loss(
                    z_tilde, labels, dim, gamma=self.gamma, factor=self.delta
                )
        # add regularization loss
        loss += reg_loss

        # log values
        if flag:
            self.writer.add_scalar(
                'loss_split/recons_loss', recons_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'loss_split/dist_loss', dist_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'loss_split/reg_loss', (reg_loss / self.gamma).item(), epoch_num
            )
            self.writer.add_scalar(
                'params/beta', self.cur_beta, epoch_num
            )
            self.writer.add_scalar(
                'params/capacity', self.cur_capacity, epoch_num
            )

        # compute accuracy
        accuracy = self.mean_accuracy(
            weights=weights, targets=score
        )

        return loss, accuracy

    def normalize_latent_attributes(self, latent_attributes):
        metadata = latent_attributes.clone().float()
        metadata = torch.div(metadata, to_cuda_variable(self.attr_norm_factors))
        return metadata

    def compute_representations(self, data_loader, num_batches=None):
        latent_codes = []
        attributes = []
        if num_batches is None:
            num_batches = 200
        for batch_id, batch in tqdm(enumerate(data_loader)):
            inputs, latent_attributes = self.process_batch_data(batch)
            _, _, _, _, z_tilde, _ = self.model(inputs, None, train=False)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            attributes.append(to_numpy(latent_attributes))
            if batch_id == num_batches:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attr_list = [
            attr for attr in self.attr_dict.keys()
        ]
        return latent_codes, attributes, attr_list

    def eval_model(self, data_loader, epoch_num=0):
        if self.writer is not None:
            # evaluation takes time due to computation of metrics
            # so we skip it during training epochs
            if epoch_num > 1 and epoch_num // 10 == 0:
                metrics = self.compute_eval_metrics()
                print(json.dumps(metrics, indent=2))
            else:
                metrics = None
        else:
            metrics = self.compute_eval_metrics()
        return metrics

    def compute_eval_metrics(self):
        """Returns the saved results as dict or computes them"""
        results_fp = os.path.join(
            os.path.dirname(self.model.filepath),
            'results_dict.json'
        )
        batch_size = 512
        _, _, gen_test = self.dataset.data_loaders(batch_size=batch_size, split=(0.70, 0.20))
        latent_codes, attributes, attr_list = self.compute_representations(gen_test)
        self.metrics.update(compute_mig(latent_codes, attributes))
        mig_factors = self.metrics["mig_factors"]
        self.metrics["mig_factors"] = {attr: mig for attr, mig in zip(attr_list, mig_factors)}
        self.metrics.update(compute_modularity(latent_codes, attributes))
        self.metrics.update(compute_sap_score(latent_codes, attributes))
        self.metrics.update(self.test_model(batch_size=batch_size))
        with open(results_fp, 'w') as outfile:
            json.dump(self.metrics, outfile, indent=2)
        return self.metrics

    def test_model(self, batch_size):
        _, _, gen_test = self.dataset.data_loaders(batch_size)
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print('Test Epoch:')
        print(
            '\tTest Loss: ', mean_loss_test, '\n'
            '\tTest Accuracy: ', mean_accuracy_test * 100
        )
        return {
            "test_loss": mean_loss_test,
            "test_acc": mean_accuracy_test,
        }

    def loss_and_acc_test(self, data_loader):
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, _ = self.process_batch_data(batch)
            # compute forward pass
            outputs, _, _, _, _, _ = self.model(
                measure_score_tensor=inputs,
                measure_metadata_tensor=None,
                train=False
            )
            # compute loss
            recons_loss = self.reconstruction_loss(
                x=inputs, x_recons=outputs
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = self.mean_accuracy(
                weights=outputs,
                targets=inputs
            )
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    @staticmethod
    def reconstruction_loss(x, x_recons):
        return Trainer.mean_crossentropy_loss(weights=x_recons, targets=x)

    @staticmethod
    def compute_reg_loss(z, labels, reg_dim, gamma, factor=1.0):
        """
        Computes the regularization loss
        """
        x = z[:, reg_dim]
        reg_loss = Trainer.reg_loss_sign(x, labels, factor=factor)
        return gamma * reg_loss
