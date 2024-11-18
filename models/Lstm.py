import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.output_size = configs.c_out
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name
        
        # Option 1: Dec-enc
        #self.lstm_dec = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        # Option 2: Pure Lstm
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)      

    def forecast(self, x, y):
        # LSTM expects input shape: (batch_size, seq_length, input_size)
        # x shape: (batch_size, seq_length, 1)
        lstm_out, hidden_states = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_size)
        lstm_out_last = lstm_out[:, -1, :]  # Extract last hidden state, shape: (batch_size, hidden_size)

        # Option 1: Enc-Dec
        '''
        h0 = torch.zeros(hidden_states[0].shape)
        c0 = torch.zeros(hidden_states[1].shape)

        # Pass the last hidden state to the greedy decoder
        h0[:, 0, :] = hidden_states[0][:, -1, :]
        c0[:, 0, :] = hidden_states[1][:, -1, :]
        '''

        # Option 2: Pure LSTM
        dec_in = y[:, -1, :]
        preds = torch.zeros(y.shape)

        # Greedy decoder
        for i in range(self.pred_len):
            # Option 1: Enc-Dec
            #output, hidden_states = self.lstm_dec(dec_in.unsqueeze(1), hidden_states) 

            # Option 2: Pure LSTM
            output, hidden_states = self.lstm(dec_in.unsqueeze(1), hidden_states) 
            dec_in = self.fc(output[:, -1, :])    # output shape: (batch_size, output_size)
            preds[:, -self.pred_len+i, :] = dec_in
        
        return preds

    def imputation(self, x):
        print(f"Imputation for LSTM to be done")
        return 0

    def anomaly_detection(self, x):
        print(f"Imputation for LSTM to be done")
        return 0

    def classification(self, x):
        print(f"Imputation for LSTM to be done")
        return 0

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out
        return None

import torch
import torch.nn as nn
import numpy as np

# Example dataset: Sine wave
def generate_sine_wave(seq_length, num_samples):
    x = np.linspace(0, seq_length * np.pi, num_samples)
    data = np.sin(x)
    return data

# Hyperparameters
SEQ_LENGTH = 50   # Length of input sequence
PRED_LENGTH = 1   # Number of steps to predict
HIDDEN_SIZE = 64  # Number of features in the hidden state
NUM_LAYERS = 2    # Number of LSTM layers
BATCH_SIZE = 32   # Batch size
EPOCHS = 10       # Number of training epochs

# Generate dataset
data = generate_sine_wave(seq_length=SEQ_LENGTH, num_samples=500)
X = []
y = []

for i in range(len(data) - SEQ_LENGTH - PRED_LENGTH + 1):
    X.append(data[i:i + SEQ_LENGTH])
    y.append(data[i + SEQ_LENGTH:i + SEQ_LENGTH + PRED_LENGTH])

X = np.array(X)  # Shape: (num_samples, SEQ_LENGTH)
y = np.array(y)  # Shape: (num_samples, PRED_LENGTH)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, SEQ_LENGTH)
y = torch.tensor(y, dtype=torch.float32)  # Shape: (num_samples, PRED_LENGTH)

# Dataset and DataLoader
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define the LSTM model
class LSTMForecasting(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecasting, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM expects input shape: (batch_size, seq_length, input_size)
        # x shape: (batch_size, seq_length, 1)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_size)
        lstm_out_last = lstm_out[:, -1, :]  # Extract last hidden state, shape: (batch_size, hidden_size)
        output = self.fc(lstm_out_last)    # output shape: (batch_size, output_size)
        return output

# Initialize the model
model = LSTMForecasting(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=PRED_LENGTH)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    for batch_X, batch_y in dataloader:
        # Reshape input to (batch_size, seq_length, input_size)
        batch_X = batch_X.unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
        
        # Forward pass
        predictions = model(batch_X)  # predictions shape: (batch_size, output_size)
        
        # Compute loss
        loss = criterion(predictions, batch_y)  # batch_y shape: (batch_size, output_size)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Inference
model.eval()
test_input = torch.tensor(data[-SEQ_LENGTH:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Shape: (1, seq_length, 1)
predicted = model(test_input)  # Shape: (1, output_size)
print("Predicted:", predicted.detach().numpy())
