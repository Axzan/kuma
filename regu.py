import torch
from . import utils


def E_L0loss(a: torch.Tensor, b: torch.Tensor, l: float,r: float,DEBUG:bool=False):
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
    
    _,prob_nonzero=utils.get_probnull(a,b,l,r,DEBUG)
    
    batch_loss=prob_nonzero.sum(-1)
    
    regloss=batch_loss.mean(0)/prob_nonzero.size(-1)
    
    return regloss

def E_FusedLassoloss(a: torch.Tensor,b: torch.Tensor,l: float,r: float,DEBUG:bool=False):
    """Computes the expectation of the fused lasso loss.

    Args:
        a (torch.Tensor): _description_
        b (torch.Tensor): _description_
        l (float): _description_
        r (float): _description_

    Returns:
        _type_: _description_
    """
    
    prob_zero, prob_nonzero = utils.get_probnull(a, b, l, r, DEBUG)

    # Ensure we have batch x sequence dims in all cases
    if prob_zero.dim() == 1:
        prob_zero = prob_zero.unsqueeze(1)
        prob_nonzero = prob_nonzero.unsqueeze(1)

    # If sequence length is 1, there are no transitions (fused lasso is 0)
    if prob_zero.size(1) < 2:
        return torch.tensor(0.0, device=prob_zero.device, dtype=prob_zero.dtype)

    z_to_nonzprob = torch.mul(prob_zero[:, :-1], prob_nonzero[:, 1:])
    nonz_to_zprob = torch.mul(prob_nonzero[:, :-1], prob_zero[:, 1:])
    switchprob = z_to_nonzprob + nonz_to_zprob

    # shapes now (B,N-1)
    batchloss = switchprob.sum(1)  # (B)
    regloss = batchloss.mean()
    return regloss

def CE_sparseconnected_loss(pred: torch.Tensor,label: torch.Tensor,gamma0: float, gamma1: float,a: torch.Tensor, b: torch.Tensor, l: float , r: float,DEBUG:bool=False):
    
    
    if DEBUG:
        assert l<0
        assert r>1
        assert gamma0>=0
        assert gamma1>=0
    
    #pred.shape : (B,nclasses)
    #label.shape : (B) ou (B,1)?
    
    batch_size=label.shape[0]
    
    label=label.view(batch_size)
    
    taskloss=torch.nn.functional.cross_entropy(pred,label,reduction="mean")
    
    reg0loss=E_L0loss(a,b,l,r,DEBUG)
    
    reg1loss=E_FusedLassoloss(a,b,l,r,DEBUG)
    
    return taskloss+gamma0*reg0loss+gamma1*reg1loss