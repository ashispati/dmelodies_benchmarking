import os
import json
import time
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import music21
from typing import Tuple
from tensorboardX import SummaryWriter

from src.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer
from src.dspritesvae.dsprites_factorvae import DspritesFactorVAE
from src.dspritesvae.image_vae_trainer import DSPRITES_ATTRIBUTES
from src.utils.helpers import to_cuda_variable, to_numpy
from src.utils.evaluation import *


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


class ImageFactorVAETrainer(DMelodiesVAETrainer):
    def __init__(
            self,
            dataset,
            model: DspritesFactorVAE,
            model_type='beta-VAE',
            lr=1e-4,
            beta=0.001,
            capacity=0.0,
            gamma=10,
            device=0,
            rand=0,
    ):
        super(ImageFactorVAETrainer, self).__init__(dataset, model, model_type, lr, beta, capacity, device, rand)
        self.attr_dict = DSPRITES_ATTRIBUTES
        self.dec_dist = 'bernoulli'
        self.warm_up_epochs = -1
        self.gamma = gamma
        self.cur_gamma = gamma
        self.trainer_config = f'_b_{self.beta}_c_{capacity}_g_{self.gamma}_r_{self.rand_seed}_'
        self.model.update_trainer_config(self.trainer_config)
        
        # reinstantiate optimizers
        self.optimizer = optim.Adam(model.VAE.parameters(), lr=lr) 
        self.D_optim = optim.Adam(model.Discriminator.parameters(), lr=1e-4, betas=(0.8, 0.9))

    # Overload trainer method
    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        if epoch_num > self.warm_up_epochs:
            # if self.anneal_iterations < self.num_iterations:
            #     if self.model_type == 'beta-VAE':
            #         self.cur_beta = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
            #     elif self.model_type == 'annealed-VAE':
            #         self.cur_beta = self.beta
            #         self.cur_capacity = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
            # self.anneal_iterations += 1
            self.cur_beta = self.beta
            self.cur_capacity = self.capacity

    # Overload trainer method
    def train_model(self, batch_size, num_epochs, log=False):
        """
        Trains the model
        :param batch_size: int,
        :param num_epochs: int,
        :param log: bool, logs epoch stats for viewing in tensorboard if TRUE
        :return: None
        """
        # set-up log parameters
        if log:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            # configure tensorboardX summary writer
            self.writer = SummaryWriter(
                logdir=os.path.join('runs/' + self.model.__repr__() + st)
            )

        # get dataloaders
        (generator_train,
         generator_val,
         _) = self.dataset.data_loaders(
            batch_size=batch_size,
            split=(0.70, 0.20)
        )
        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))

        # train epochs
        for epoch_index in range(num_epochs):
            # run training loop on training data
            self.model.train()
            returns = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True
            )
            mean_loss_train, mean_accuracy_train = returns[0]
            mean_D_loss_train, mean_D_accuracy_train = returns[1]

            # run evaluation loop on validation data
            self.model.eval()
            returns = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False
            )
            mean_loss_val, mean_accuracy_val = returns[0]
            mean_D_loss_val, mean_D_accuracy_val = returns[1]

            self.eval_model(
                data_loader=generator_val,
                epoch_num=epoch_index,
            )

            # log parameters
            if log:
                # log value in tensorboardX for visualization
                self.writer.add_scalar('loss/train', mean_loss_train, epoch_index)
                self.writer.add_scalar('loss/valid', mean_loss_val, epoch_index)
                self.writer.add_scalar('acc/train', mean_accuracy_train, epoch_index)
                self.writer.add_scalar('acc/valid', mean_accuracy_val, epoch_index)
                self.writer.add_scalar('loss/D_train', mean_D_loss_train, epoch_index)
                self.writer.add_scalar('loss/D_valid', mean_D_loss_val, epoch_index)
                self.writer.add_scalar('acc/D_train', mean_D_accuracy_train, epoch_index)
                self.writer.add_scalar('acc/D_valid', mean_D_accuracy_val, epoch_index)

            # print epoch stats
            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_train,
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_val,
                'mean_accuracy_val': mean_accuracy_val
            }
            self.print_epoch_stats(**data_element)

            # save model
            self.model.save()

    # overload trainer method
    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        mean_D_loss = 0
        mean_D_accuracy = 0
        for batch_num, batch in tqdm(enumerate(data_loader)):
            # update training scheduler
            if train:
                self.update_scheduler(epoch_num)

            # process batch data
            batch_1, batch_2 = self.process_batch_data(batch)

            # zero the gradients
            self.zero_grad()

            # compute loss for batch
            vae_loss, accuracy, D_z = self.loss_and_acc_for_batch(
                batch_1, epoch_num, batch_num, train=train
            )

            # compute backward and step if train
            if train:
                vae_loss.backward(retain_graph=True)
                # self.plot_grad_flow()
                self.step()

            # compute Discriminator loss
            D_loss, D_acc = self.loss_and_acc_for_batch_D(
                batch_2, D_z, epoch_num, batch_num, train=train
            )

            if train:
                self.D_optim.zero_grad()
                D_loss.backward()
                self.D_optim.step()

            # log batch_wise:
            self.writer.add_scalar(
                'batch_wise/vae_loss', vae_loss.item(), self.global_iter
            )
            self.writer.add_scalar(
                'batch_wise/D_loss', D_loss.item(), self.global_iter
            )

            # compute mean loss and accuracy
            mean_loss += to_numpy(vae_loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)
            mean_D_loss += to_numpy(D_loss.mean())
            mean_D_accuracy += to_numpy(D_acc.mean())

            if train:
                self.global_iter += 1

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        mean_D_loss /= len(data_loader)
        mean_D_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        ), (
            mean_D_loss,
            mean_D_accuracy
        )

    def process_batch_data(self, batch, test=False):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        inputs, labels = batch
        b = inputs.size(0)
        inputs = to_cuda_variable(inputs)
        labels = to_cuda_variable(labels)
        if test:
            return inputs, labels
        else:
            if b%2 != 0:
                b -= 1
            batch_1 = (inputs[:b//2], labels[:b//2])
            batch_2 = (inputs[b//2:b], labels[b//2:b])
            return (batch_1, batch_2)

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        """
        Computes the VAE loss and accuracy for the batch
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
        inputs, latent_attributes = batch

        # perform forward pass of model
        outputs, z_dist, prior_dist, z_tilde, z_prior = self.model(inputs)

        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(inputs, outputs, self.dec_dist)

        # compute KLD loss
        if self.model_type == 'beta-VAE':
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.beta, c=0.0)
            dist_loss = torch.nn.functional.relu(dist_loss - self.cur_capacity)
        elif self.model_type == 'annealed-VAE':
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=self.cur_capacity)
        else:
            raise ValueError('Invalid Model Type')

        # compute TC loss
        D_z = self.model.forward_D(z_tilde)
        tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        # add loses
        loss = recons_loss + dist_loss + self.cur_gamma*tc_loss

        # compute and add regularization loss if needed
        # log values
        if flag:
            self.writer.add_scalar(
                'loss_split/recons_loss', recons_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'loss_split/dist_loss', dist_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'params/beta', self.cur_beta, epoch_num
            )
            self.writer.add_scalar(
                'loss_split/vae_tc_loss', tc_loss.item(), epoch_num
            )

        # compute accuracy
        accuracy = self.mean_accuracy(
            weights=torch.sigmoid(outputs),
            targets=inputs
        )

        return loss, accuracy, D_z

    def loss_and_acc_for_batch_D(self, batch, D_z, epoch_num=None, batch_num=None, train=True):
        """
        Computes the Discriminator loss and accuracy for the batch
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
        inputs, latent_attributes = batch

        batch_size = inputs.shape[0]

        ones = torch.ones(batch_size, dtype=torch.long).cuda()
        zeros = torch.zeros(batch_size, dtype=torch.long).cuda()

        z_tilde = self.model.encode(inputs)
            
        z_pperm = permute_dims(z_tilde).detach()
        D_z_pperm = self.model.forward_D(z_pperm)
        D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

        if flag:
            self.writer.add_scalar(
                'loss_split/d_tc_loss', D_tc_loss.item(), epoch_num
            )

        soft_D_z = F.softmax(D_z, 1)[:, :1].detach()
        soft_D_z_pperm = F.softmax(D_z_pperm, 1)[:, :1].detach()
        D_acc = ((soft_D_z >= 0.5).sum() + (soft_D_z_pperm < 0.5).sum()).float()
        D_acc /= 2*batch_size

        return D_tc_loss, D_acc

    def _extract_relevant_attributes(self, attributes):
        attr_list = [
            attr for attr in self.attr_dict.keys() if attr != 'color'
        ]
        attr_idx_list = [
            self.attr_dict[attr] for attr in attr_list
        ]
        attr_labels = attributes[:, attr_idx_list]
        return attr_labels, attr_list

    def compute_representations(self, data_loader, num_batches=None):
        latent_codes = []
        attributes = []
        if num_batches is None:
            num_batches = 200
        for batch_id, batch in tqdm(enumerate(data_loader)):
            inputs, labels = self.process_batch_data(batch, test=True)
            _, _, _, z_tilde, _ = self.model(inputs)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            attributes.append(to_numpy(labels))
            if batch_id == num_batches:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0).astype('int32')
        attributes, attr_list = self._extract_relevant_attributes(attributes)
        return latent_codes, attributes, attr_list

    def loss_and_acc_test(self, data_loader):
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, _ = self.process_batch_data(batch, test=True)
            inputs = to_cuda_variable(inputs)
            # compute forward pass
            outputs, _, _, _, _ = self.model(inputs)
            # compute loss
            recons_loss = self.reconstruction_loss(
                inputs, outputs, self.dec_dist
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = self.mean_accuracy(
                weights=torch.sigmoid(outputs),
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
    def reconstruction_loss(x, x_recons, dist):
        batch_size = x.size(0)
        if dist == 'bernoulli':
            recons_loss = F.binary_cross_entropy_with_logits(
                x_recons, x, size_average=False,
            ).div(batch_size)
        elif dist == 'gaussian':
            x_recons = torch.sigmoid(x_recons)
            recons_loss = F.mse_loss(
                x_recons, x, size_average=False
            ).div(batch_size)
        else:
            raise AttributeError("invalid dist")
        return recons_loss

    @staticmethod
    def mean_accuracy(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return float, accuracy
        """
        predictions = torch.zeros_like(weights)
        predictions[weights >= 0.5] = 1
        binary_targets = torch.zeros_like(targets)
        binary_targets[targets >= 0.5] = 1
        correct = predictions == binary_targets
        acc = torch.sum(correct.float()) / binary_targets.view(-1).size(0)
        return acc

    @staticmethod
    def mean_accuracy_pred(pred_labels, gt_labels):
        correct = pred_labels.long() == gt_labels.long()
        return torch.sum(correct.float()) / pred_labels.size(0)
