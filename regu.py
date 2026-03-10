import torch
from . import utils




def E_L0loss(a: torch.Tensor, b: torch.Tensor, l: float,r: float, sequencedim : int = -1, batchdim : int = 0 ):
    """ Computes the expectation of the L0 loss of the HardKuma samples (number of non zero samples) and averages it by batch.

    Args:
        l<0
        r>1
        (a > 0).all()
        (b > 0).all()
        a.shape==b.shape
        a (torch.Tensor): 1st parameter tensor of arbitrary shape.
        b (torch.Tensor): 2nd parameter tensor of arbitrary shape.
        l (float): leftmost point of the stretching.
        r (float): rightmost point of the stretching.

    Returns:
        loss (torch.Tensor): value of the expected L0 loss averaged by batch
    """
    
    _,prob_nonzero=utils.get_probnull(a,b,l,r)
    
    batch_loss=prob_nonzero.sum(dim=sequencedim)
    
    regloss=batch_loss.mean(dim=batchdim)
    
    return regloss

def E_FusedLassoloss(a: torch.Tensor,b: torch.Tensor,l: float,r: float):
    """Computes the expectation of the fused lasso loss.

    Args:
        a (torch.Tensor): _description_
        b (torch.Tensor): _description_
        l (float): _description_
        r (float): _description_

    Returns:
        _type_: _description_
    """
    
    prob_zero,prob_nonzero=utils.get_probnull(a,b,l,r)
    
    #shape : (B,N) 
    
    z_to_nonzprob=torch.mul(prob_zero[:,:-1],prob_nonzero[:,1:])
    
    nonz_to_zprob=torch.mul(prob_nonzero[:,:-1],prob_zero[:,1:])
    
    switchprob=z_to_nonzprob+nonz_to_zprob
    
    #shapes au dessus (B,N-1)
    
    batchloss=switchprob.sum(dim=1)         #(B)
    
    regloss=batchloss.mean()
    
    return regloss

def CE_sparseconnected_loss(pred: torch.Tensor,label: torch.Tensor,gamma0: float, gamma1: float,a: torch.Tensor, b: torch.Tensor, l: float , r: float):
    
    assert l<0
    assert r>1
    assert gamma0>=0
    assert gamma1>=0
    
    #pred.shape : (B,nclasses)
    #label.shape : (B) ou (B,1)?
    
    batch_size=label.shape[0]
    
    label=label.view(batch_size)
    
    taskloss=torch.nn.functional.cross_entropy(pred,label,reduction="mean")
    
    reg0loss=E_L0loss(a,b,l,r)
    
    reg1loss=E_FusedLassoloss(a,b,l,r)
    
    return taskloss+gamma0*reg0loss+gamma1*reg1loss