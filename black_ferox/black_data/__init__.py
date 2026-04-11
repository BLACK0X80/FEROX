import random
import math
import queue
import threading



class BlackDataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, black_idx):
        raise NotImplementedError

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackTensorDataset(BlackDataset):
    def __init__(self, *black_tensors):
        self.black_tensors = black_tensors
        if len(black_tensors) > 0:
            self.black_size = len(black_tensors[0]) if hasattr(black_tensors[0], '__len__') else 0

    def __len__(self):
        return self.black_size

    def __getitem__(self, black_idx):
        return tuple(black_t[black_idx] for black_t in self.black_tensors)


class BlackIterableDataset:
    def __iter__(self):
        raise NotImplementedError

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackRandomSampler:
    def __init__(self, black_dataset, black_replacement=False, black_num_samples=None):
        self.black_dataset = black_dataset
        self.black_replacement = black_replacement
        self.black_num_samples = black_num_samples or len(black_dataset)

    def __iter__(self):
        black_n = len(self.black_dataset)
        if self.black_replacement:
            for _ in range(self.black_num_samples):
                yield random.randint(0, black_n - 1)
        else:
            black_indices = list(range(black_n))
            random.shuffle(black_indices)
            for black_idx in black_indices[:self.black_num_samples]:
                yield black_idx

    def __len__(self):
        return self.black_num_samples


class BlackSequentialSampler:
    def __init__(self, black_dataset):
        self.black_dataset = black_dataset

    def __iter__(self):
        return iter(range(len(self.black_dataset)))

    def __len__(self):
        return len(self.black_dataset)


class BlackDistributedSampler:
    def __init__(self, black_dataset, black_num_replicas, black_rank,
                 black_shuffle=True, black_seed=0):
        self.black_dataset = black_dataset
        self.black_num_replicas = black_num_replicas
        self.black_rank = black_rank
        self.black_shuffle = black_shuffle
        self.black_seed = black_seed
        self.black_epoch = 0
        self.black_total_size = math.ceil(len(black_dataset) / black_num_replicas) * black_num_replicas
        self.black_num_samples = self.black_total_size // black_num_replicas

    def __iter__(self):
        if self.black_shuffle:
            black_g = random.Random(self.black_seed + self.black_epoch)
            black_indices = list(range(len(self.black_dataset)))
            black_g.shuffle(black_indices)
        else:
            black_indices = list(range(len(self.black_dataset)))

        black_indices += black_indices[: self.black_total_size - len(black_indices)]
        black_indices = black_indices[self.black_rank: self.black_total_size: self.black_num_replicas]
        return iter(black_indices)

    def __len__(self):
        return self.black_num_samples

    def black_set_epoch(self, black_epoch):
        self.black_epoch = black_epoch


def black_default_collate(black_batch):
    black_elem = black_batch[0]
    if isinstance(black_elem, (int, float)):
        return black_batch
    if isinstance(black_elem, str):
        return black_batch
    if isinstance(black_elem, (list, tuple)):
        black_transposed = list(zip(*black_batch))
        return [black_default_collate(black_samples) for black_samples in black_transposed]
    if isinstance(black_elem, dict):
        return {black_key: black_default_collate([black_d[black_key] for black_d in black_batch]) for black_key in black_elem}
    return black_batch


class BlackDataLoader:
    def __init__(self, black_dataset, black_batch_size=1, black_shuffle=False,
                 black_num_workers=0, black_pin_memory=False, black_drop_last=False,
                 black_collate_fn=None, black_sampler=None, black_prefetch_factor=2):
        self.black_dataset = black_dataset
        self.black_batch_size = black_batch_size
        self.black_shuffle = black_shuffle
        self.black_num_workers = black_num_workers
        self.black_pin_memory = black_pin_memory
        self.black_drop_last = black_drop_last
        self.black_collate_fn = black_collate_fn or black_default_collate
        self.black_sampler = black_sampler
        self.black_prefetch_factor = black_prefetch_factor

    def __iter__(self):
        if self.black_sampler is not None:
            black_indices = list(self.black_sampler)
        elif self.black_shuffle:
            black_indices = list(range(len(self.black_dataset)))
            random.shuffle(black_indices)
        else:
            black_indices = list(range(len(self.black_dataset)))

        black_batches = []
        for black_i in range(0, len(black_indices), self.black_batch_size):
            black_batch_indices = black_indices[black_i: black_i + self.black_batch_size]
            if self.black_drop_last and len(black_batch_indices) < self.black_batch_size:
                continue
            black_batch = [self.black_dataset[black_idx] for black_idx in black_batch_indices]
            black_batches.append(self.black_collate_fn(black_batch))

        if self.black_num_workers > 0 and self.black_prefetch_factor > 0:
            black_q = queue.Queue(maxsize=self.black_prefetch_factor)
            black_done = threading.Event()

            def black_producer():
                for black_batch in black_batches:
                    if black_done.is_set():
                        break
                    black_q.put(black_batch)
                black_q.put(None)

            black_thread = threading.Thread(target=black_producer, daemon=True)
            black_thread.start()

            while True:
                black_item = black_q.get()
                if black_item is None:
                    break
                yield black_item

            black_done.set()
            black_thread.join(timeout=1.0)
        else:
            for black_batch in black_batches:
                yield black_batch

    def __len__(self):
        black_n = len(self.black_dataset)
        if self.black_drop_last:
            return black_n // self.black_batch_size
        return math.ceil(black_n / self.black_batch_size)

from black_ferox.black_data.black_upgrades import ( # noqa: E402
    BlackStreamingDataset as BlackStreamingDataset,
    BlackPackedDataset as BlackPackedDataset,
    BlackDatasetHub as BlackDatasetHub,
    BlackSyntheticDataGenerator as BlackSyntheticDataGenerator,
)
