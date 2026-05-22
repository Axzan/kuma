import torch
# Fonctions de répartitions de la distrib originale et modifiée

def frep_kuma(x : float, a : torch.Tensor, b : torch.Tensor,DEBUG:bool=False): 
    """Returns the value of the Kuma repartition function at x for all parameter values in the tensors a and b.

    Args and requirements:
        (a > 0).all()
        (b > 0).all()
        a.shape==b.shape
        x (float): Value at which the repartition function should be taken
        a (torch.Tensor): 1st parameter tensor of arbitrary shape.
        b (torch.Tensor): 2nd parameter tensor of arbitrary shape.

    Returns:
        frep (torch.Tensor): Repartition function values of same shape as the input tensors.
    """
    if DEBUG:
        assert a.shape==b.shape
        assert (a > 0).all().item()
        assert (b > 0).all().item()
    
    original_shape=a.shape
    
    aflat=a.flatten()
    bflat=b.flatten()
    
    frepflat= 1 - (1 - x**aflat)**bflat
    
    frep=frepflat.reshape(original_shape)
    
    return frep

def frep_hardkuma(x : float,a : torch.Tensor,b : torch.Tensor,l : float,r : float,DEBUG:bool=False):
    """Returns the value of the (l,r)-HardKuma repartition function at x for all parameter values in the input tensors a and b.

    Args and requirements:
        l<0
        r>1
        (a > 0).all()
        (b > 0).all()
        a.shape==b.shape
        x (float): Value at which the repartition function should be taken.
        a (torch.Tensor): 1st parameter tensor of arbitrary shape.
        b (torch.Tensor): 2nd parameter tensor of arbitrary shape.
        l (float): leftmost point of the stretching.
        r (float): rightmost point of the stretching.

    Returns:
        frep (torch.Tensor): Repartition function values of same shape as the input tensors.
    """
    if DEBUG:
        assert a.shape==b.shape
        assert l<0
        assert r>1
        assert (a > 0).all().item()
        assert (b > 0).all().item()
    
    original_shape=a.shape
    
    d=a.device
    
    if x<0:
        
        return torch.zeros(original_shape).to(d)
    
    elif x < (1-l)/(r-l):
        
        return frep_kuma((x-l)/(r-l),a,b,DEBUG)
    
    else:
        
        return torch.ones(original_shape).to(d)

def get_probnull(a : torch.Tensor,b: torch.Tensor,l: float,r: float,DEBUG:bool=False):
    """Returns the probability of an (a,b,l,r)-HardKuma sample being 0 for every value of the parameter tensors.

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
        prob_null,prob_non_null: zero and nonzero probability tensor of same shape as the input tensors.
    """
    d=a.device
    
    prob_null=frep_hardkuma(0,a,b,l,r,DEBUG)
    
    prob_nonnull=torch.ones(prob_null.shape).to(d)-prob_null
    
    return prob_null,prob_nonnull