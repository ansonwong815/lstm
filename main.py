# %%
import numpy
import torch
from torch import nn
import pandas as pd
from datetime import datetime
import time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# %%
df = pd.read_csv('MSFT.csv')
# %%
df = df.drop("Date", axis=1)
df = df.drop("Volume", axis=1)
torch.manual_seed(0)


# %%
def absolute_maximum_scale(series):
    return series / series.mean()


for i in df:
    df[i] = df[i].astype("float64")
    df[i] = absolute_maximum_scale(df[i])
    df[i].fillna(df[i].mean())
# %%
label = np.zeros(len(df["Close"]) - 1)
for i in range(1, len(df["Close"])):
    label[i - 1] = df["Close"][i]
df.drop(df.tail(1).index, inplace=False)
# %%
torch.manual_seed(0)
input_size = 5
hidden_size = 128
num_layers = 2
output_size = 1
learning_rate = 0.00001
num_epochs = 100
data_length = len(label)
seq_len = 128
stride_size = 4
batch_size = 32
device = torch.device("mps")
writer = SummaryWriter(filename_suffix=f"batch_size_{batch_size},lr_{learning_rate}",
                       comment=f"batch_size_{batch_size},lr_{learning_rate}")


class CustomDataset(Dataset):
    def __init__(self):
        self.seq_len = seq_len
        self.data = [(torch.tensor(df.iloc[p:p + seq_len].to_numpy(), device=device, dtype=torch.float32),
                      torch.tensor(label[p + seq_len], dtype=torch.float32, device=device)) for p in
                     range(0, data_length - stride_size, stride_size) if p + seq_len < len(df) - 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


torch.manual_seed(0)
dataset = CustomDataset()
training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=350)
train_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True, drop_last=True)
print("done")

torch.manual_seed(0)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# %%
torch.manual_seed(0)
network = LSTM().to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
criterion = nn.MSELoss().to(device)
# torch.save((network.state_dict(), optimizer.state_dict()), "blankmodel/blank.pth")
state = torch.load("blankmodel/blank.pth")
network.load_state_dict(state[0])
optimizer.load_state_dict(state[1])
optimizer.param_groups[0]['lr'] = learning_rate

# %%
print(len(train_data_loader))
# %%
start_time = time.time()
test_time = 0
_iter = 0
torch.manual_seed(0)
for epoch in range(num_epochs):
    network.train()
    train_loss = 0
    for batch_X, batch_y in train_data_loader:
        _iter += 1
        batch_y = batch_y.to(dtype=batch_X.dtype)
        optimizer.zero_grad()
        outputs = network(batch_X)
        batch_y = batch_y.reshape(outputs.shape)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        writer.add_scalar("Train Loss/Data", loss.item(), _iter * batch_size)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss / len(train_data_loader):.4f}")
    writer.add_scalar("Train Loss/Epoch", train_loss / len(train_data_loader), epoch + 1)
    writer.add_scalar("Epoch/Time", epoch + 1, int(time.time() - start_time - test_time))
    if (epoch + 1) % 5 == 0:
        start_test = time.time()
        network.eval()
        totalloss = 0
        with torch.no_grad():
            for X_test_tensor, y_test_tensor in test_data_loader:
                y_pred = network(X_test_tensor)
                y_pred = y_pred.reshape(y_test_tensor.shape)
                test_loss = criterion(y_pred, y_test_tensor)
                totalloss += test_loss.item()
        print(f"Test Loss: {(totalloss / len(test_data_loader)):.4f}")
        writer.add_scalar("Test Loss/Epoch", totalloss / len(test_data_loader), epoch + 1)
        test_time += time.time() - start_test
