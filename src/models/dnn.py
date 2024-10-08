from copy import deepcopy

from sklearn.preprocessing import StandardScaler
import torch 
from torch import nn 

class IOCLinear(nn.Module):
    def __init__(self, in_dim, label_weights, out_dim=22, layers=6, first_hidden=2048, saved_scalar=None):
        '''
        Torch implementation of Ramiro's Keras models
        '''
        super().__init__()

        def layer(d, out_d=None):
            if out_d is None:
                out_d = d//2

            return nn.Sequential(
                nn.Linear(d, out_d), 
                nn.LayerNorm(out_d), 
                nn.Dropout(),
                nn.ReLU()
            )
        
        self.net = nn.Sequential(
            layer(in_dim, out_d=first_hidden), 
            *[layer(first_hidden//(2**i)) for i in range(0,layers-1)],
            nn.Sequential(
                nn.Linear(first_hidden//(2**(layers-1)), out_dim),
                nn.Softmax(dim=1)
            )
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=label_weights, label_smoothing=1e-5)

        if saved_scalar is None:
            self.scale_mean = None 
            self.scale_std = None
        else:
            self.scale_mean = saved_scalar[0]
            self.scale_std = saved_scalar[1]

        self.args = (in_dim, label_weights)
        self.kwargs = dict(out_dim=out_dim, layers=layers, first_hidden=first_hidden)

    def preprocess_fit(self, x):
        self.scale_mean = x.mean(0, keepdim=True).detach()
        self.scale_std = x.std(0, unbiased=False, keepdim=True).detach()
        self.kwargs['saved_scalar'] = (self.scale_mean, self.scale_std)

    def preprocess(self, x):    
        x = x - self.scale_mean
        x = x / (self.scale_std+1e-8)

        return x 

    def forward(self, x, targets):
        preds = self.inference(x)
        return self.loss_fn(preds, targets)

    def inference(self, x):
        x = self.preprocess(x)
        return self.net(x)
    
    def predict(self, x):
        self.eval()
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.inference(x).numpy()
    
    def save(self):
        return (self.args, self.kwargs, deepcopy(self.state_dict()))