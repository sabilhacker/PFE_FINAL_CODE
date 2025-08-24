from typing import Optional , Sequence, List
import numpy as np
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from qiskit_machine_learning.kernels import (
    TrainableKernel,
)

class DataBatcher:
    """
    A class used to batch dataset and labels.
    """

    def __init__(self, dataset, labels):
        """
        Initialize a DataBatches object with the input dataset and corresponding labels.

        Args:
            dataset (numpy array): A numpy array of shape (num_samples, num_features) containing the input dataset.
            labels (numpy array): A numpy array of shape (num_samples,) containing the corresponding labels for the dataset.

        Returns:
            None
        """
        self.dataset = dataset
        self.labels = labels
        self.num_samples = len(dataset)
        self.unique_labels, self.label_counts = np.unique(labels, return_counts=True)

    def balanced_batches(self, batch_size, shuffle=False):
        """
        Generate a list of balanced batches, where each batch contains the same number of samples from each label.

        Args:
            batch_size (int): The desired size of each batch.
            shuffle (bool): if True, shuffle batches.

        Returns:
            batches (List): a list of batches where each batch is a tuple containing the batch data and corresponding labels.
        """
        if batch_size > self.num_samples:
            raise ValueError(
                f"Batch size {batch_size} is larger than the dataset size {self.num_samples}"
            )
        if batch_size > 2 * np.min(self.label_counts):
            raise ValueError(
                f"Batch size {batch_size} is 2x larger than the smallest label size {np.min(self.label_counts)}"
            )
        samples_per_label = batch_size // len(self.unique_labels)
        batches = []
        for _ in range(self.num_samples // batch_size):
            batch_data = []
            batch_labels = []
            for l in self.unique_labels:
                label_indices = np.where(self.labels == l)[0]
                if shuffle:
                    np.random.shuffle(label_indices)
                if samples_per_label > len(label_indices):
                    batch_indices = label_indices
                else:
                    batch_indices = label_indices[:samples_per_label]
                batch_data.append(self.dataset[batch_indices])
                batch_labels.append(self.labels[batch_indices])
            batch_data = np.concatenate(batch_data, axis=0)
            batch_labels = np.concatenate(batch_labels, axis=0)
            batches.append((batch_data, batch_labels))
        return batches

    def imbalanced_batches(self, batch_size, keep_ratio=False, shuffle=False):
        """
        Generate a list of imbalanced batches, where each batch may contain a different number of samples from each label.

        Args:
            batch_size (int): The desired size of each batch.
            keep_ratio (bool): If True, maintain the same relative frequency of each label as in the original dataset.
                            If False, use the absolute frequency of each label to determine the number of samples per label.
            shuffle (bool): If True, shuffle batches.

        Returns:
            batches (list): a list of batches where each batch is a tuple containing the batch data and corresponding labels.
        """
        if batch_size > self.num_samples:
            raise ValueError(
                f"Batch size {batch_size} is larger than the dataset size {self.num_samples}"
            )
        if keep_ratio:
            # calculate the number of samples per label based on the relative label frequencies
            label_freqs = self.label_counts / np.sum(self.label_counts)
            samples_per_label = np.round(batch_size * label_freqs).astype(int)
        else:
            # calculate the number of samples per label based on the absolute label frequencies
            samples_per_label = np.round(
                batch_size * self.label_counts / np.sum(self.label_counts)
            ).astype(int)
        batches = []
        for _ in range(self.num_samples // batch_size):
            batch_data = []
            batch_labels = []
            for l, num_samples in zip(self.unique_labels, samples_per_label):
                label_indices = np.where(self.labels == l)[0]
                if shuffle:
                    np.random.shuffle(label_indices)
                batch_indices = label_indices[:num_samples]
                batch_data.append(self.dataset[batch_indices])
                batch_labels.append(self.labels[batch_indices])
            batch_data = np.concatenate(batch_data, axis=0)
            batch_labels = np.concatenate(batch_labels, axis=0)
            batches.append((batch_data, batch_labels))
        return batches
class BatchedSVCLoss(SVCLoss):
    r"""
    This class provides a kernel loss function for classification tasks by fitting an ``SVC`` model
    from scikit-learn, extended for use with batches. Given training samples, :math:`x_{i}`, with binary labels, :math:`y_{i}`,
    and a kernel, :math:`K_{θ}`, parameterized by values, :math:`θ`, the loss is defined as:

    .. math::

        SVCLoss = \sum_{i} a_i - 0.5 \sum_{i,j} a_i a_j y_{i} y_{j} K_θ(x_i, x_j)

    where :math:`a_i` are the optimal Lagrange multipliers found by solving the standard SVM
    quadratic program. Note that the hyper-parameter ``C`` for the soft-margin penalty can be
    specified through the keyword args.

    Minimizing this loss over the parameters, :math:`θ`, of the kernel is equivalent to maximizing a
    weighted kernel alignment, which in turn yields the smallest upper bound to the SVM
    generalization error for a given parameterization.

    See https://arxiv.org/abs/2105.03406 for further details.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        sub_kernel_size: Optional[int] = None,
        minibatch_size: Optional[int] = 1,
        shuffle: bool = False,
        balanced_batch: bool = False,
        keep_ratio: bool = True,
        encoder=None,
        **kwargs,
    ):
        """
        Args:
            data (np.ndarray): The data to evaluate the loss on.
            labels (np.ndarray): The corresponding labels for the data.
            sub_kernel_size (int, optional): The size of the sub-kernel batches to split the data into. If not provided,
                the entire data set is used in a single batch.
            shuffle (bool, optional): Whether to shuffle the data before splitting into batches. Default is False.
            balanced_batch (bool, optional): Whether to use balanced or imbalanced batching. Default is False.
            encoder (torch.nn): An instance to optionally reduce dimension before calculating loss
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor within
                      SVCLoss evaluation.
        """
        super().__init__(**kwargs)
        # Split data into batches
        self.sub_kernel_size = sub_kernel_size
        bg = DataBatcher(data, labels)
        self.minibatch_size = minibatch_size
        if self.sub_kernel_size == None:
            self.batches = [data, labels]
        elif balanced_batch:
            self.batches = bg.balanced_batches(sub_kernel_size, shuffle=shuffle)
        else:
            self.batches = bg.imbalanced_batches(
                sub_kernel_size, keep_ratio=keep_ratio, shuffle=shuffle
            )

        self.idx = 0
        self.epoch = 0
        self.encoder = encoder
        self.loss_arr = []

    def evaluate(
        self,
        parameters: Sequence[float],
        quantum_kernel: TrainableKernel,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Wrapper function for loss evaluation with batches of data. If sub_kernel_size is None, it will execute SVCLoss() on full dataset.

        Args:
            parameter_values (Sequence[float]): The parameter values to evaluate the loss with.
            quantum_kernel (TrainableKernel): The quantum kernel to use for evaluation.
        Returns:
            loss (float): the loss value for the given parameters and quantum kernel.
        """
        if self.sub_kernel_size == None:
            if type(self.encoder) != type(None):
                weights = parameters[: self.encoder.num_weights]
                variational_params = parameters[self.encoder.num_weights :]
                self.encoder.set_weights(weights)
                encoded_data = self.encoder.encode(data)
                return super().evaluate(variational_params, quantum_kernel, encoded_data, labels)
            else:
                loss = super().evaluate(parameters, quantum_kernel, data, labels)
                self.loss_arr.append(loss)
                return loss

        if self.idx + self.minibatch_size > len(self.batches):
            self.idx = 0
            self.epoch += 1

        mini_batch = self.batches[self.idx : self.idx + self.minibatch_size]
        # Evaluate the loss for each batch and accumulate the total loss
        total_loss = 0
        i = self.idx
        for batch_data, batch_labels in mini_batch:
            if type(self.encoder) != type(None):
                weights = parameters[: self.encoder.num_weights]
                variational_params = parameters[self.encoder.num_weights :]
                self.encoder.set_weights(weights)
                batch_data = self.encoder.encode(batch_data)
            else:
                variational_params = parameters
            loss = super().evaluate(variational_params, quantum_kernel, batch_data, batch_labels)
            total_loss += loss
            i += 1
        self.idx += self.minibatch_size
        self.loss_arr.append(total_loss / self.minibatch_size)
        param_loss = total_loss / self.minibatch_size
        return param_loss