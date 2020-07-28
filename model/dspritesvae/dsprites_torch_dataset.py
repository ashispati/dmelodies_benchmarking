import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(
        np.transpose(npimg, (1, 2, 0)), interpolation='nearest'
    )


class DspritesDataset:
    def __init__(self, seed=0):
        self.kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
        self.root_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        ))
        self.data_path = os.path.join(
            self.root_dir, 'dsprites-dataset', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        )
        self.dataset = None
        self.seed = seed
        np.random.seed(seed)

    def load_dataset(self):
        data = np.load(self.data_path, encoding='bytes')
        images = data["imgs"]
        images = np.expand_dims(images, axis=1).astype('float32')
        latent_values = data["latents_values"].astype('float32')
        a = np.c_[
            images.reshape(len(images), -1),
            latent_values.reshape(len(latent_values), -1)
        ]
        images2 = a[:, :images.size // len(images)].reshape(images.shape)
        latent_values2 = a[:, images.size // len(images):].reshape(latent_values.shape)
        np.random.shuffle(a)
        self.dataset = TensorDataset(
            torch.from_numpy(images2),
            torch.from_numpy(latent_values2)
        )

    def data_loaders(self, batch_size, split=(0.70, 0.20)):
        """
        Returns three data loaders obtained by splitting
        self.dataset according to split
        :param batch_size:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        :param split:
        :return:
        """
        assert sum(split) < 1

        if self.dataset is None:
            self.load_dataset()

        num_examples = len(self.dataset)
        a, b = split
        train_dataset = TensorDataset(
            *self.dataset[: int(a * num_examples)]
        )
        val_dataset = TensorDataset(
            *self.dataset[int(a * num_examples):int((a + b) * num_examples)]
        )
        eval_dataset = TensorDataset(
            *self.dataset[int((a + b) * num_examples):]
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **self.kwargs
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_dl, val_dl, eval_dl
