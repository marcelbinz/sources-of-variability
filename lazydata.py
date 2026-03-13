import torch
from torch.utils.data import IterableDataset
import polars as pl
import numpy as np


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
        last_chunk = np.array(max(self.participants) % self.participant_subsets)[
            np.newaxis
        ]
        chunklengths = np.concatenate([full_chunks, last_chunk])
        uppers = np.cumsum(chunklengths)
        lowers = np.concatenate([[0], uppers[:-1]])

        for idx, _ in enumerate(chunklengths):
            # fetch a chunk lazily
            participants_use = self.participants[lowers[idx] : uppers[idx]]
            chunk = self.lf.filter(
                pl.col("participant").is_in(participants_use)
            ).collect()

            df_x = chunk.select([self.col_pid + self.cols_x + self.col_y_shifted])
            df_y = chunk.select([self.col_pid + self.col_y])

            groups_x = (
                df_x.groupby(self.col_pid)
                .agg([pl.col(c) for c in self.cols_x + self.col_y_shifted])
                .collect()
            )
            groups_y = df_y.groupby(self.col_pid).agg(pl.col(self.col_y)).collect()

            seqs_x = [
                torch.tensor(row, dtype=torch.float32)
                for row in groups_x.select(self.cols_x + self.col_y_shifted).to_numpy()
            ]
            seqs_y = [
                torch.tensor(row, dtype=torch.long)
                for row in groups_y.select(self.col_y).to_numpy()
            ]

            X_3d = torch.nn.utils.rnn.pad_sequence(seqs_x, batch_first=True)
            y_3d = torch.nn.utils.rnn.pad_sequence(seqs_y, batch_first=True)
            lengths = torch.tensor([len(s) for s in seqs_x])
            mask = torch.arange(X_3d.size(1))[None, :] < lengths[:, None]

            # yield row by row (or yield batches if you prefer)
            # for xi, yi, m in zip(X_3d, y_3d, mask):
            #     yield xi, yi, m
            yield X_3d, y_3d, mask
