import numpy as np
import torch


def collate(samples):
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


class IndexedDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        return self.dataset[int(self.indices[position])]


def batch_indices(size, batch_size, seed, step):
    generator = np.random.default_rng([seed, step])
    return generator.integers(0, size, size=batch_size)


def random_batches(dataset, batch_size, seed, start_step=0):
    step = start_step
    while True:
        indices = batch_indices(len(dataset), batch_size, seed, step)
        yield collate([dataset[int(index)] for index in indices])
        step += 1


def ordered_batches(dataset, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        yield collate([dataset[int(index)] for index in chunk])


def evenly_spaced_indices(size, count):
    return np.linspace(0, size - 1, min(count, size)).astype(int)
