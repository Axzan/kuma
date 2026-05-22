import torch

# Fonctions de sampling différentiable

def diffsamp_Kuma(a : torch.Tensor,b : torch.Tensor,DEBUG:bool=False):
    """Samples in a differentiable way from a (a,b)-Kuma distribution.

    Args and requirements:
        (a > 0).all()
        (b > 0).all()
        a.shape==b.shape
        a (torch.Tensor): 1st parameter tensor of arbitrary shape.
        b (torch.Tensor): 2nd parameter tensor of arbitrary shape.

    Returns:
        k : Kuma sample of same shape as the input tensors.
    """
    if DEBUG:
        assert a.shape==b.shape
        assert (a >= 0).all().item()
        assert (b >= 0).all().item()
    
    eps=10**(-3)
    
    original_shape=a.shape
    
    d=a.device
    
    aflat=torch.flatten(a)
    bflat=torch.flatten(b)
    
    num_samples=aflat.shape
    
    u=torch.rand(num_samples).to(d)
    
    k=(1-(1-u)**(1/(bflat+eps))**(1/(aflat+eps)))
    
    return k.reshape(original_shape)


def diffsamp_HKuma(a : torch.Tensor,b : torch.Tensor,l : float, r : float,DEBUG : bool=False):
    """Samples in a almost everywhere differentiable way from an (a,b,l,r)-HardKuma distribution.

    Args and requirements:
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
        h (torch.Tensor): HardKuma sample of same shape as the input tensors.
    """
    
    if DEBUG:
        assert l<0
        assert r>1
    
    k=diffsamp_Kuma(a,b,DEBUG)
    
    t=l+(r-l)*k
    
    h=torch.clamp(t,min=0,max=1)
    
    return h