import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# home-grown
import utils as ut

# TODO different data loaders for different conditions
# TODO add logging
# TODO positional encoding
# load all four dataframes and concat into one df
# each data frame from a different 2020 month (i.e., march, april, june, november)
df_itc_all = ut.load_and_concat_raw_data()
df_itc_all.drop("Unnamed: 0", axis=1, inplace=True)


batch_size = 32
num_epochs = 100
lr = 3e-4
num_hidden = 256
num_layers = 6
d_model = 64
num_head = 8


# create pytorch dataset
num_features = 4
num_targets = 1
num_participants =  # TODO
# careful: assumes all participants have equal number of trials
X = torch.zeros(num_participants, num_trials, num_features)
Y = torch.zeros(num_participants, num_trials, 1)
Xprime = torch.cat([X, torch.cat([torch.zeros(num_participants, 1, 1), Y[:, :-1, :], dim = 1])])

# create torch data loader
dataset = TensorDataset(Xprime, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# TODO make sure that causal attention mask is used
model = TransformerDecoder(num_input=env.num_dims, num_output=1, num_hidden=num_hidden,
                           num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=num_trials, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        print(batch_x, batch_y)
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(-1), batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
