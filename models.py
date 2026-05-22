import torch
from . import sample

#B : batch size, N le nombre de tokens maximum, H la dimension de l'encodage
### Shape (B,N,H)


class KumaParams(torch.nn.Module):
    
    def __init__(self,H,hlstm=128,hlinear=64):
        
        super().__init__()
        
        self.RNN_encoder=torch.nn.LSTM(H,hlstm,bidirectional=True,batch_first=True)
        self.layera=torch.nn.Linear(2*hlstm,hlinear)
        self.layerb=torch.nn.Linear(2*hlstm,hlinear)
        self.finala=torch.nn.Linear(hlinear,1)
        self.finalb=torch.nn.Linear(hlinear,1)
        # Initialize biases negative so exp(final(...)) is small initially,
        # encouraging zero selections before learning starts.
        torch.nn.init.constant_(self.finala.bias, -3.0)
        torch.nn.init.constant_(self.finalb.bias, -3.0)
    
    def forward(self,x):
        #x.shape : (B,N,H)
        
        enc,(hn,cn)=self.RNN_encoder(x)
        #enc.shape : (B,N,2*hlstm)
        
        x_a = torch.relu(self.layera(enc))
        x_b = torch.relu(self.layerb(enc))
        #shapes : (B,N,hlinear)
        
        out_a = torch.exp(self.finala(x_a)).squeeze(-1)
        out_b = torch.exp(self.finalb(x_b)).squeeze(-1)
        #shapes : (B,N) even when N==1
        #fixed the squeeze that could collapse into a and b into 1D tensors.
        
        return out_a, out_b
    


# Modèle sélecteur - calcul des paramètres Kuma, Sampling différentiable puis masking : (N,H) -> (N,H)

class KumaSelector(torch.nn.Module):
    
    def __init__(self, H, l, r, hlstm=128, hlinear=64,DEBUG=False):
        super().__init__()
        self.l=l
        self.r=r
        self.Kumaparams=KumaParams(H,hlstm,hlinear)
        self._DEBUG=DEBUG

    def forward(self,x):
        #x.shape : (B,N,H)
        
        a,b=self.Kumaparams(x)
        #shape : (B,N) , (B,N)
        
        mask=sample.diffsamp_HKuma(a,b,self.l,self.r,self._DEBUG)
        #shape : (B,N)

        return mask,a,b