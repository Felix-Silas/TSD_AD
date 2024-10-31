'''This is mostly adapted from GitHub repo https://github.com/manigalati/usad.,
original paper was published at KDD 2020 at https://dl.acm.org/doi/10.1145/3394486.3403392, 
titled "USAD: UnSupervised Anomaly Detection on Multivariate Time Series".
Please also check the authors' original paper and implementation for reference.'''

'''One small modification here is that I added the function testing_threshold() based on 
the contamination rate already known, which returns the threshold for interpreting anomaly scores.'''

from utils import *

import torch
import torch.nn as nn

device = get_default_device()

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size/4))
        self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
        self.linear3 = nn.Linear(int(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w
    
class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        loss1 = 1/n * torch.mean((batch - w1)**2) + (1-1/n)*torch.mean((batch - w3) ** 2)
        loss2 = 1/n * torch.mean((batch - w2)**2) - (1-1/n)*torch.mean((batch - w3) ** 2)
        
        return loss1, loss2
    
    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        loss1 = 1/n * torch.mean((batch - w1)**2) + (1-1/n)*torch.mean((batch - w3)**2)
        loss2 = 1/n * torch.mean((batch - w2)**2) - (1-1/n)*torch.mean((batch - w3)**2)

        return {'val_loss1': loss1, 'val_loss2' : loss2}
    
    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        batch_losses2 = [x['val_loss2'] for x in outputs]

        epoch_loss1 = torch.stack(batch_losses1).mean()
        epoch_loss2 = torch.stack(batch_losses2).mean()

        return {'val_epoch_loss1' : epoch_loss1.item(), 'val_epoch_loss2' : epoch_loss2.item()}
    
    def epoch_end(self, epoch, result):
        print('epoch [{}], val_epoch_loss1: {:.4f}, val_epoch_loss2: {:.4f}'.format(epoch, result['val_epoch_loss1'], result['val_epoch_loss2']))

def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch, device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def training(epochs, model, train_loader, val_loader):
    history = []

    optimizer1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()))

    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)
            
            #Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            #Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch + 1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def training_scores(model, train_loader, val_loader = None, alpha = 0.5, beta = .5):
    if val_loader is None:
        train_val = train_loader

    else:
        train_val = [d for dl in [train_loader, val_loader] for d in dl]

    results = []
    
    for [batch] in train_val:
        batch = to_device(batch, device)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))
        results.append(alpha * torch.mean((batch-w1)**2, dim = 1) + beta * torch.mean((batch - w2) ** 2, dim = 1))

    return results

def testing_scores(model, test_loader, alpha = .5, beta = .5):
    results = []

    for [batch] in test_loader:
        batch = to_device(batch, device)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))

        results.append(alpha * torch.mean((batch - w1) ** 2, dim = 1) + beta * torch.mean((batch - w2) ** 2, dim = 1))

    return results

def testing_threshold(model, test_loader, alpha = .5, beta = .5, contamination = 0.1): # 상위 10%의 점수를 이상치로 하겠다
    results = testing_scores(model, test_loader, alpha = .5, beta = .5)
    score_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().numpy(),
                                 results[-1].flatten().detach().numpy()])
    return np.sort(score_pred)[int(len(score_pred) * (1- contamination))]