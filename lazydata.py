import torch
from torch.utils.data import IterableDataset
import polars as pl
import numpy as np
from tqdm import tqdm


class PolarsLazyDataset(IterableDataset):
    def __init__(
        self,
        lazyframe,
        participants,
        col_pid,
        col_y,
        cols_x,
        col_y_shifted,
        participant_subsets=1000,
    ):
        self.lf = lazyframe
        self.participants = participants
        self.col_pid = col_pid
        self.col_y = col_y
        self.cols_x = cols_x
        self.col_y_shifted = col_y_shifted
        self.participant_subsets = participant_subsets
        self.n_chunks = len(self.participants) // self.participant_subsets + 1

    def __len__(self):
        return self.n_chunks

    def __iter__(self):

        full_chunks = np.repeat(self.participant_subsets, (self.n_chunks - 1))
        last_chunk = np.array(len(self.participants) % self.participant_subsets)[
            np.newaxis
        ]
        chunklengths = np.concatenate([full_chunks, last_chunk])
        uppers = np.cumsum(chunklengths)
        lowers = np.concatenate([[0], uppers[:-1]])

        feature_cols = self.cols_x + self.col_y_shifted

        for idx, _ in enumerate(tqdm(chunklengths)):
            # fetch a chunk lazily
            participants_use = self.participants[lowers[idx] : uppers[idx]]
            chunk = self.lf.filter(
                pl.col(self.col_pid).is_in(participants_use)
            ).collect()

            # Group X: list columns per participant
            groups_x = chunk.group_by(self.col_pid).agg(
                [pl.col(c) for c in feature_cols]
            )

            # Group Y: list column per participant
            groups_y = chunk.group_by(self.col_pid).agg(pl.col(self.col_y))

            seqs_x = []
            for row in groups_x.iter_rows(named=True):
                # Extract list columns in order
                arrays = [
                    np.asarray(row[c]).astype(float).reshape(-1) for c in feature_cols
                ]
                # Stack into shape (n_trials, n_features)
                mat = np.column_stack(arrays)
                seqs_x.append(torch.tensor(mat, dtype=torch.float32))

            seqs_y = []
            for row in groups_y.iter_rows(named=True):
                arr = np.array(row[self.col_y[0]])
                seqs_y.append(torch.tensor(arr, dtype=torch.long))

            X_3d = torch.nn.utils.rnn.pad_sequence(seqs_x, batch_first=True)
            y_3d = torch.nn.utils.rnn.pad_sequence(seqs_y, batch_first=True)
            lengths = torch.tensor([len(s) for s in seqs_x])
            mask = torch.arange(X_3d.size(1))[None, :] < lengths[:, None]

            # yield row by row (or yield batches if you prefer)
            # for xi, yi, m in zip(X_3d, y_3d, mask):
            #     yield xi, yi, m
            yield X_3d, y_3d, mask
