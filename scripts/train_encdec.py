import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import os

ts_len = 2000 # synthetic time series length
hidden_size = 80 # lstm hidden size
test_ds_len = 500 # test dataset size
epochs = 3000 # training epochs
ts_history_len = 200 # sliding window size
ts_target_len = 60 # prediction length

directory = "/users/jdvillegas/repos/Time-Series-Library-Fork/dataset/local-methane-data-diego/2ndVisit/txt/"
#directory = "/home/julian/Documents/local-methane-data-diego/2ndVisit/txt/"

all_data = pd.DataFrame()

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        df = df.iloc[:, [0,1]]
        all_data = pd.concat([all_data, df])

all_data.sort_values(by=all_data.columns[0], inplace=True)
#all_data.to_csv("merged_methane_meas.csv", index=False, header=True)

all_data['hour'] = all_data['DATE'].apply(lambda x: int(float(x.split(":")[0])))
all_data['min'] = all_data['DATE'].apply(lambda x: int(float(x.split(":")[1])))
all_data['sec'] = all_data['DATE'].apply(lambda x: float(x.split(":")[2]))

all_data['hour_sin'] = np.sin(2 * np.pi * all_data['hour']/24)
all_data['hour_cos'] = np.cos(2 * np.pi * all_data['hour']/24)
all_data['min_sin'] = np.sin(2 * np.pi * all_data['min']/60)
all_data['min_cos'] = np.cos(2 * np.pi * all_data['min']/60)
all_data['sec_sin'] = np.sin(2 * np.pi * all_data['sec']/60)
all_data['sec_cos'] = np.cos(2 * np.pi * all_data['sec']/60)

all_data.drop(columns=['hour', 'min', 'sec'], inplace=True)

all_data.head(20)

all_data.drop(columns=['DATE'], inplace=True)
all_data

all_data.drop(columns=['hour_cos', 'hour_sin', 'min_sin', 'min_cos', 'sec_sin',	'sec_cos'], inplace=True)
all_data

myts = all_data.values.tolist()
myts = np.array(myts).squeeze().tolist()

def generate_ts(len):
    tf = 80 * np.pi
    t = np.linspace(0., tf, len)
    y = np.sin(t) + 0.8 * np.cos(.5 * t) + np.random.normal(0., 0.3, len) + 2.5
    return y.tolist()

def sliding_window(ts, features, target_len = 1):
    # target_len : how many steps (samples) to predict
    X = []
    Y = []

    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len):i - target_len])
        Y.append(ts[i - target_len:i])

    return X, Y

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)

    def forward(self, x):
        flat = x.view(x.shape[0], x.shape[1], self.input_size)
        out, h = self.lstm(flat)
        return out, h
    
class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size = 1, num_layers = 1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.lstm(x.unsqueeze(0), h)
        y = self.linear(out.squeeze(0))
        return y, h

class EncoderDecoder(nn.Module):

    def __init__(self, hidden_size, input_size = 1, output_size = 1):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = Decoder(input_size = input_size, hidden_size = hidden_size, output_size = output_size)

    def train_model(
            self, train, target, epochs, target_len, method = 'recursive',
            tfr = 0.5, lr = 0.01, dynamic_tf = False
    ):
        losses = np.full(epochs, np.nan)
        optimizer = optim.Adam(self.parameters(), lr = lr)
        criterion = nn.MSELoss()

        for e in range(epochs):
            predicted = torch.zeros(target_len, train.shape[1], train.shape[2])
            optimizer.zero_grad()
            _, enc_h = self.encoder(train)

            dec_in = train[-1, :, :]
            dec_h = enc_h

            if method == 'recursive':
                for t in range(target_len):
                    dec_out, dec_h = self.decoder(dec_in, dec_h)
                    predicted[t] = dec_out
                    dec_in = dec_out

            if method == 'teacher_forcing':
                # use teacher forcing
                if random.random() < tfr:
                    for t in range(target_len):
                        dec_out, dec_h = self.decoder(dec_in, dec_h)
                        predicted[t] = dec_out
                        dec_in = target[t, :, :]
                # predict recursively
                else:
                    for t in range(target_len):
                        dec_out, dec_h = self.decoder(dec_in, dec_h)
                        predicted[t] = dec_out
                        dec_in = dec_out

            if method == 'mixed_teacher_forcing':
                # predict using mixed teacher forcing
                for t in range(target_len):
                    dec_out, dec_h = self.decoder(dec_in, dec_h)
                    predicted[t] = dec_out
                    # predict with teacher forcing
                    if random.random() < tfr:
                        dec_in = target[t, :, :]
                    # predict recursively
                    else:
                        dec_in = dec_out

            loss = criterion(predicted, target)
            loss.backward()
            optimizer.step()

            losses[e] = loss.item()

            if e % 10 == 0:
                print(f'Epoch {e}/{epochs}: {round(loss.item(), 4)}')

            # dynamic teacher forcing
            if dynamic_tf and tfr > 0:
                tfr = tfr - 0.02

        return losses

    def predict(self, x, target_len):
        y = torch.zeros(target_len, x.shape[1], x.shape[2])

        _, enc_h = self.encoder(x)
        dec_in = x[-1, :, :]
        dec_h = enc_h

        for t in range(target_len):
            dec_out, dec_h = self.decoder(dec_in, dec_h)
            y[t] = dec_out
            dec_in = dec_out

        return y

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

X, Y = sliding_window(myts, ts_history_len, ts_target_len)
ds_len = len(X)

def to_tensor(data):
    
    return torch.tensor(data = data)\
        .unsqueeze(2)\
        .transpose(0, 1).float()
    
    '''
    return torch.tensor(data = data)\
        .transpose(0, 1).float()
    '''

x_train = to_tensor(X[:ds_len - test_ds_len])
y_train = to_tensor(Y[:ds_len - test_ds_len])
x_test = to_tensor(X[ds_len - test_ds_len:])
y_test = to_tensor(Y[ds_len - test_ds_len:])

model = EncoderDecoder(hidden_size = hidden_size, input_size=1)

model.train()
model.train_model(x_train, y_train, epochs, ts_target_len,
                  method = 'mixed_teacher_forcing',
                  tfr = .05, lr = .005)

model.eval()
predicted = model.predict(x_test, ts_target_len)

fig, ax = plt.subplots(nrows = 10, ncols = 1)
fig.set_size_inches(7.5, 15)
for col in ax:
    r = random.randint(0, test_ds_len)

    # Get one of the test sequences
    in_seq = x_test[:, r, 0].view(-1).tolist()

    # True multi-step values
    target_seq = y_test[:, r, 0].view(-1).tolist()
    pred_seq = predicted[:, r, :].view(-1).tolist()
    x_axis = range(len(in_seq) + len(target_seq))
    col.set_title(f'Test Sample: {r}')
    col.axis('off')
    col.plot(x_axis[:], in_seq + target_seq, color = 'blue')
    col.plot(x_axis[len(in_seq):],
             pred_seq,
             label = 'predicted',
             color = 'orange',
             linestyle = '--',
             linewidth = 3)
    col.vlines(len(in_seq), 0, np.max(in_seq), color = 'grey')
    col.legend(loc = "upper right")

plt.show()