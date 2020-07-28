import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from model.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer
from model.dspritesvae.dsprites_vae import DspritesVAE
from model.utils.helpers import to_cuda_variable, to_numpy
from model.utils.evaluation import *
from model.utils.trainer import Trainer


DSPRITES_ATTRIBUTES = {
    "color": 0,
    "shape": 1,
    "scale": 2,
    "orientation": 3,
    "posx": 4,
    "posy": 5
}


class ImageVAETrainer(DMelodiesVAETrainer):
    def __init__(
            self,
            dataset,
            model: DspritesVAE,
            model_type='beta-VAE',
            lr=1e-4,
            beta=0.001,
            capacity=0.0,
            device=0,
            rand=0,
    ):
        super(ImageVAETrainer, self).__init__(dataset, model, model_type, lr, beta, capacity, device, rand)
        self.attr_dict = DSPRITES_ATTRIBUTES
        self.dec_dist = 'bernoulli'

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        inputs, labels = batch
        inputs = to_cuda_variable(inputs)
        labels = to_cuda_variable(labels)
        return inputs, labels

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
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.beta, c=self.cur_capacity)
        else:
            raise ValueError('Invalid Model Type')

        # add losses
        loss = recons_loss + dist_loss

        # log values
        if flag:
            self.writer.add_scalar(
                'loss_split/recons_loss', recons_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'loss_split/dist_loss', dist_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'params/beta', self.beta, epoch_num
            )
            self.writer.add_scalar(
                'params/capacity', self.cur_capacity, epoch_num
            )

        # compute accuracy
        accuracy = self.mean_accuracy(
            weights=torch.sigmoid(outputs),
            targets=inputs
        )

        return loss, accuracy

    def _extract_relevant_attributes(self, attributes):
        attr_list = [
            attr for attr in self.attr_dict.keys() if attr != 'color'
        ]
        attr_idx_list = [
            self.attr_dict[attr] for attr in attr_list
        ]
        attr_labels = attributes[:, attr_idx_list]
        attr_labels[:, 1] = attr_labels[:, 1] * 10.0 - 5
        attr_labels[:, 2] = attr_labels[:, 2] * (40.0 - 1) / (2 * np.pi)
        attr_labels[:, 3] = attr_labels[:, 3] * (32.0 - 1)
        attr_labels[:, 4] = attr_labels[:, 4] * (32.0 - 1)
        attr_labels = attr_labels.astype('int32')
        return attr_labels, attr_list

    def compute_representations(self, data_loader, num_batches=None):
        latent_codes = []
        attributes = []
        if num_batches is None:
            num_batches = 200
        for batch_id, batch in tqdm(enumerate(data_loader)):
            inputs, labels = self.process_batch_data(batch)
            _, _, _, z_tilde, _ = self.model(inputs)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            attributes.append(to_numpy(labels))
            if batch_id == num_batches:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attributes, attr_list = self._extract_relevant_attributes(attributes)
        return latent_codes, attributes, attr_list

    def plot_latent_interpolations(self, attr_str, dim, num_points=10):
        x1 = torch.linspace(-4, 4.0, num_points)
        _, _, data_loader = self.dataset.data_loaders(batch_size=1)
        for sample_id, batch in tqdm(enumerate(data_loader)):
            if sample_id in [0, 1, 2]:
                inputs, labels = self.process_batch_data(batch)
                inputs = to_cuda_variable(inputs)
                recons, _, _, z, _ = self.model(inputs)
                recons = torch.sigmoid(recons)
                z = z.repeat(num_points, 1)
                z[:, dim] = x1.contiguous()
                outputs = torch.sigmoid(self.model.decode(z))
                # save interpolation
                save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'latent_interpolations_{attr_str}_{sample_id}.png'
                )
                save_image(
                    outputs.cpu(), save_filepath, nrow=num_points, pad_value=1.0
                )
                # save original image
                org_save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'original_{sample_id}.png'
                )
                save_image(
                    inputs.cpu(), org_save_filepath, nrow=1, pad_value=1.0
                )
                # save reconstruction
                recons_save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'recons_{sample_id}.png'
                )
                save_image(
                    recons.cpu(), recons_save_filepath, nrow=1, pad_value=1.0
                )
            if sample_id == 5:
                break

    def loss_and_acc_test(self, data_loader):
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, _ = self.process_batch_data(batch)
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
