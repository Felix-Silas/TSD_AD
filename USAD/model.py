import torch
import torch.nn as nn
import torch.nn.init as init
from utils import *
from sklearn.manifold import TSNE

device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True) # inplace = True
    
    # 가중치 초기화
    init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
    init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
    init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')
    
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

        # 가중치 초기화
        init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')
    
    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w
    

class USAD_model(nn.Module):
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
        #loss1 = 1/n * torch.mean((batch - w1)**2) + (1-1/n) * torch.mean((batch - w3)**2)
        #loss2 = 1/n * torch.mean((batch - w2)**2) - (1-1/n) * torch.mean((batch - w3)**2)
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
        return loss1, loss2
    
    def validation_step(self, batch, n):
        with torch.no_grad():
            z = self.encoder(batch)
            w1 = self.decoder1(z)
            w2 = self.decoder2(z)

            w3 = self.decoder2(self.encoder(w1))
            #loss1 = 1/n * torch.mean((batch - w1)**2) + (1-1/n) * torch.mean((batch - w3)**2)
            #loss2 = 1/n * torch.mean((batch - w2)**2) - (1-1/n) * torch.mean((batch - w3)**2)
            loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
            loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
            
        return {'val_loss1': loss1, 'val_loss2': loss2}
    
    def cal_epoch_loss(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
    def print_loss(self, epoch, result):
        print('Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}'.format(epoch + 1, result['val_loss1'], result['val_loss2']))

def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.cal_epoch_loss(outputs)

def training(epochs, model, train_loader, val_loader, optimizer = torch.optim.Adam):
    history = []
    optimizer1 = optimizer(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = optimizer(list(model.encoder.parameters()) + list(model.decoder2.parameters()))

    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad() # 누적되기 떄문에 이전 gradient 초기화

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad() 

        result = evaluate(model, val_loader, epoch + 1)
        model.print_loss(epoch, result)
        history.append(result)
        
    return history

def testing(model, test_loader, alpha = .5, beta = .5):
    results = []

    with torch.no_grad():
        for [batch] in test_loader:
            batch = to_device(batch, device)
            w1 =model.decoder1(model.encoder(batch))
            w2 = model.decoder2(model.encoder(w1))
            results.append(alpha * torch.mean((batch - w1)**2, axis = 1) + beta * torch.mean((batch - w2)**2, axis = 1))
        return results
    

def training_scores(model, train_loader, val_loader = None, alpha =.5, beta = .5):
    if val_loader is None:
        trainval = train_loader
    
    else:
        trainval = [d for dl in [train_loader, val_loader] for d in dl]
    
    results = []
    for [batch] in trainval:
        batch = to_device(batch, device)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))
        results.append(alpha * torch.mean((batch - w1)**2, dim = 1)+beta * torch.mean((batch - w2)**2, dim = 1))
    return results

def testing_scores(model, test_loader, alpha=.5, beta=.5):
  results=[]

  for [batch] in test_loader:
      batch = to_device(batch, device)
      w1 = model.decoder1(model.encoder(batch))
      w2 = model.decoder2(model.encoder(w1))
      results.append(alpha * torch.mean((batch - w1)**2, dim = 1)+beta * torch.mean((batch - w2)**2, dim = 1))

  return results

# 상위 contamination만큼 이상치로 판단
def testing_threshold(model, test_loader, alpha=.5, beta=.5, contamination=0.1):

  results = testing_scores(model, test_loader, alpha=.5, beta=.5)

  score_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().numpy(), 
                                         results[-1].flatten().detach().numpy()])
                                         
  return np.sort(score_pred)[int(len(score_pred) * (1 - contamination))]