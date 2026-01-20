import torch
from .importance import get_normalizing_constant
from collections.abc import Iterable
from copy import deepcopy

def get_masks():
    return {'X': torch.tensor([1,-1]),
            'Y': torch.tensor([-1,1]),
            },{'X': torch.tensor([1,0]),
            'Y': torch.tensor([0,1]),
            }

def score_forward(backbone, x, t=None, mask=None, std=None):
    """
    Perform score inference on the input data.

    Args:
        x (torch.Tensor): Concatenated variables.
        t (torch.Tensor, optional): The time t. 
        mask (torch.Tensor, optional): The mask data.
        std (torch.Tensor, optional): The standard deviation to rescale the network output.

    Returns:
        torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
    """
    t = t.expand(t.shape[0],mask.size(-1)) 
    
    marg = (- mask).clip(0, 1) ## max <0 
    cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
        
    t = t * (1- cond)  + 0.0 * cond
    t = t* (1-marg) + 1 * marg

    return backbone(x, t, std=std)


def infer_scores(backbone, xy_t,t, xy_0, std_w, marg_masks, cond_mask, x_shape, minde_type):
    if minde_type=="c":

        marg_x = xy_t.clone()
        marg_x[:,x_shape[0]:] = 0
        cond_x = xy_t.clone()
        cond_x[:,x_shape[0]:] = xy_0[:,x_shape[0]:]
        
        s_marg = - score_forward(backbone, marg_x, t=t, mask=marg_masks['X'], std=std_w).detach()
        s_cond = - score_forward(backbone, cond_x, t=t, mask=cond_mask['X'], std=std_w).detach()

        s_marg = s_marg[:, :x_shape[0]]
        s_cond = s_cond[:, :x_shape[0]]
        return s_marg, s_cond

    elif minde_type=="j":

        s_joint = - score_forward(backbone, xy_t, t=t, std=std_w, mask=torch.ones_like(marg_masks['X'])).detach()

        cond_x = xy_t.clone()
        cond_x[:, x_shape[0]:] = xy_0[:, x_shape[0]:]
        cond_y = xy_t.clone()
        cond_y[:, :x_shape[0]] = xy_0[:, :x_shape[0]]

        s_cond_x = - score_forward(backbone, cond_x, t=t, mask=cond_mask['X'], std=std_w).detach() ##S(X|Y)
        s_cond_y = - score_forward(backbone, cond_y, t=t, mask=cond_mask['Y'], std=std_w).detach() ##S(Y|X)

        return s_joint, s_cond_x[:, :x_shape[0]], s_cond_y[:, x_shape[0]:]

def mi_cond(s_marg ,s_cond, g, importance_sampling):

    M = g.shape[0] 
    const = get_normalizing_constant((1,)).to(s_marg.device)

    s_marg = s_marg.view(M,-1)
    s_cond = s_cond.view(M,-1)

    # raise UserWarning(f"Some shapes: s_marg={s_marg.shape}, s_cond={s_cond.shape}, g={g.shape}")
    
    if importance_sampling:
        mi = const *0.5* ((s_marg - s_cond  )**2).sum()/ M
    else:
        try:
            mi = 0.5* (g**2*(s_marg - s_cond )**2).sum()/ M
        except:
            raise ValueError(f"Shapes g={g.shape} s_marg={s_marg.shape} s_cond={s_cond.shape}")
        
    return mi.item()

def mi_joint( s_joint ,s_cond_x,s_cond_y, g ,importance_sampling):
  
    
    M = g.shape[0] 
    s_cond = torch.cat([s_cond_x,s_cond_y],dim=1)
    s_cond = s_cond.view(M,-1)
    s_joint = s_joint.view(M,-1)

    if importance_sampling:
        const = get_normalizing_constant((1,)).to(g.device)
        mi = const *0.5* ((s_joint - s_cond  )**2).sum()/ M
    else:
        mi = 0.5 * (g**2*(s_joint - s_cond )**2).sum()/ M
    return mi.item()

def get_mutinfo_step_fn(sde, importance_sampling, minde_type, eps):

    marg_masks, cond_masks = get_masks()

    def mutinfo_step_fn(backbone, x, y):
        marg_masks['X'] = marg_masks['X'].to(x.device)
        cond_masks['X'] = cond_masks['X'].to(x.device)
        marg_masks['Y'] = marg_masks['Y'].to(y.device)
        cond_masks['Y'] = cond_masks['Y'].to(y.device)
        xy_0 = torch.cat((x, y), dim=1).float()
        M = xy_0.shape[0]
        mis = []
        for _ in range(10):
            if importance_sampling:
                t = (sde.sample_importance_sampling_t(
                    shape=(M, 1)))
            else:
                t = ((sde.T - eps) * torch.rand((M, 1)) + eps)
            t = t.to(xy_0.device)
            _, g = sde.sde(t)
            xy_t, _, mean, std = sde.sample(xy_0, t=t)
            
            std_w = None if importance_sampling else std 

            if minde_type == "c":
                s_marg, s_cond = infer_scores(backbone, xy_t, t, xy_0, std_w, marg_masks, cond_masks, x.shape[1:], minde_type)
                mi = mi_cond(s_marg=s_marg,s_cond=s_cond,g=g,importance_sampling=importance_sampling)
                
            elif minde_type=="j":
                s_joint, s_cond_x,s_cond_y = infer_scores(backbone, xy_t,t, xy_0, std_w, marg_masks, cond_masks, x.shape[1:], minde_type)
                mi = mi_joint(s_joint=s_joint,
                                s_cond_x=s_cond_x,
                                s_cond_y=s_cond_y,g=g,importance_sampling=importance_sampling)
            mis.append(mi)

        mi = torch.tensor(mis).mean()
        return mi

    return mutinfo_step_fn