import numpy as np
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, encoder_path, signal_shape = 100):
        super(Encoder, self).__init__()
        self.encoder_path = encoder_path
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(input_size = self.signal_shape, hidden_size = 20, num_layers = 1, bidirectional = True)
        self.dense = nn.Linear(in_features = 40, out_features = 20)
    
    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x, (hn, cn) = self.lstm(x) # hn, cn: hidden/cell state
        x = self.dense(x)

        return (x)
    
class Decoder(nn.Module):
    def __init__(self, decoder_path, signal_shape = 100):
        super(Decoder, self).__init__()
        self.decoder_path = decoder_path
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(input_size = 20, hidden_size = 64, num_layers = 2, bidirectional = True)
        self.dense = nn.Linear(in_features = 128, out_features = self.signal_shape)
        
    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return (x)
    
class CriticX(nn.Module):
    def __init__(self, critic_x_path, signal_shape = 100):
        super(CriticX, self).__init__()
        self.critic_x_path = critic_x_path
        self.signal_shape = signal_shape
        self.dense1 = nn.Linear(in_features = self.signal_shape, out_features = 20)
        self.dense2 = nn.Linear(in_features = 20, out_features = 1)

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x = self.dense1(x)
        x = self.dense2(x)
        return (x)
    
class CriticZ(nn.Module):
    def __init__(self, critic_z_path):
        super(CriticZ, self).__init__()
        self.critic_z_path = critic_z_path
        self.dense1 = nn.Linear(in_features = 20, out_features = 1)

    def forward(self, x):
        x = self.dense1(x)
        return (x)