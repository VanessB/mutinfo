import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

def cartesian_power(tensor, n):
    """
    Compute the Cartesian power of a tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        n (int): The number of times to take the Cartesian product of the tensor with itself.
    
    Returns:
        torch.Tensor: The Cartesian power of the input tensor.
    """
    grids = [tensor] * n
    meshgrids = torch.meshgrid(*grids, indexing='ij')
    cartesian_product = torch.stack(meshgrids, dim=-1).reshape(-1, n)
    return cartesian_product

def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @dim.setter
    def dim(self, value):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass

    def gather_transition(self, i, j, sigma):
        assert i.shape == j.shape, f"Shapes of i and j must match, got {i.shape} and {j.shape}"
        trans = self.transition(i, sigma)
        trans = torch.gather(trans, -1, j[...,None])
        return trans


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)
    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass

    def get_p_values(self, p, x):
        """
        Returns the values of p given x.
        p is expected of shape (alphabet_size, ..., alphabet_size)
        With alphabet_size repeated num_tokens times.
        x is expected of shape (..., num_tokens)
        """
        old_x_shape = x.shape
        alphabet_size = self.dim
        num_tokens = x.shape[-1]
        p = p.reshape(-1)
        x = x.reshape(-1,num_tokens)
        index_map = torch.arange(start=num_tokens-1, end=-1, step=-1, device=p.device)
        index_map = alphabet_size**index_map
        index_map = index_map[None,:].expand(x.shape[0],-1)
        new_x = (x * index_map).sum(dim=-1)
        # raise UserWarning(f"x examples: {x[:5]}, new_x examples: {new_x[:5]}, index_map examples: {index_map[:5]}")
        values = torch.gather(p, -1, new_x)
        values = values.reshape(*old_x_shape[:-1])
        return values
    
    def get_prod_ptok(self, x_to, x_from, sigma):
        assert x_to.shape == x_from.shape, f"Shapes of x_to and x_from must match, got {x_to.shape} and {x_from.shape}"
        sigma = sigma.reshape(-1)
        old_x_shape = x_to.shape
        num_tokens = x_to.shape[-1]
        alphabet_size = self.dim
        x_to = x_to.reshape(-1,num_tokens)
        x_from = x_from.reshape(-1,num_tokens)
        assert x_to.shape[0] == sigma.shape[0], f"x_to and x_from should match sigma in non token dims, got {x_to.shape} and {sigma.shape}"
        token_indeces = torch.arange(alphabet_size, device=x_to.device)
        token_indeces = token_indeces[None,None,:].expand(x_to.shape[0],num_tokens,-1)
        from_indeces = token_indeces.unsqueeze(-1).expand(-1,-1,-1,alphabet_size)
        to_indeces = token_indeces.unsqueeze(-2).expand(-1,-1,alphabet_size,-1)
        trans_probs = self.gather_transition(from_indeces, to_indeces, sigma.reshape(-1,1)).squeeze(-1)
        prod = torch.ones_like(sigma)
        batch_size = trans_probs.shape[0]
        for t in range(num_tokens):
            prod *= trans_probs[torch.arange(batch_size),t,x_to[:,t],x_from[:,t]]
        
        return prod.reshape(*old_x_shape[:-1])

    def get_pt(self, p, sigma):
        assert p is not None, "p must be provided"
        if p.shape[0] == 1:
            p = p.squeeze(0)
        num_tokens = len(p.shape)-1
        sequences = cartesian_power(torch.arange(self.dim, device=p.device), num_tokens)
        num_sequences = len(sequences)
        sequences = sequences.unsqueeze(0).expand(sigma.shape[0],-1,-1)
        # raise UserWarning(f"Sequences (shape={sequences.shape}): {sequences[:5]}")
        to_sequences = sequences.unsqueeze(-2).expand(-1,-1,num_sequences,-1)
        from_sequences = sequences.unsqueeze(1).expand(-1,num_sequences,-1,-1)

        # raise UserWarning(f"Sequences examples: {sequences[:5]}, to_sequences examples: {to_sequences[:5]}, from_sequences examples: {from_sequences[:5]}")

        p_cond_prod = self.get_prod_ptok(to_sequences, from_sequences, sigma.reshape(-1,1,1).expand(-1,num_sequences,num_sequences))
        p0 = self.get_p_values(p, from_sequences)

        # raise UserWarning(f"from_sequences examples: {from_sequences[:5]}, p0 examples: {p0[:5]}")

        p_prod = p_cond_prod * p0
        p_t = p_prod.sum(dim=-1)
        p_t = p_t.reshape(-1,*p.shape)

        return p_t
    
    def get_analytic_score(self, x, p, sigma):
        """
        Computes the score function given sigma.
        """

        assert x.shape[0] == sigma.shape[0], "sigma must match x for batch size"

        p_t = self.get_pt(p,sigma)
        p_t = p_t.squeeze(-1)
        if len(x.shape) == 2 and len(p_t.shape) == 3:
            p_t = p_t.squeeze(1)
        p_shape = p_t.shape
        batch_size = p_shape[0]
        alphabet_size = p_shape[1]
        sequence_length = x.shape[1]

        i = torch.arange(x.shape[0], device=x.device)
        indeces = tuple([i] + [x[:, j] for j in range(sequence_length)])

        den = p_t[indeces]
        # raise UserWarning(f"den shape is {den.shape}, x examples: {x[:5]}, p_t examples: {p_t[:5]}, den examples: {den[:5]}")
        den = den[...,None,None].expand(-1,sequence_length,alphabet_size)

        num = torch.empty((batch_size, sequence_length, alphabet_size)).to(p_t)  # Shape: (d0, t, j)

        # Iterate over t (the replaced dimension)
        for t in range(sequence_length):
            for a in range(alphabet_size):
                num[:,t,a] = p_t[tuple([i] + [x[:, j] if j != t else a for j in range(sequence_length)])]
        
        score = num / den
        # raise UserWarning(f"num shape is {num.shape}, p_t examples: {p_t[:5]}, x examples: {x[:5]}, score examples: {score[:5]}, den examples: {den[:5]}, num examples: {num[:5]}")

        return score
    
    def score_divergence(self, score_p, score_q, dsigma, x):

        x = x.unsqueeze(-1)

        log_score_p = score_p.log()
        score_p = torch.scatter(score_p, -1, x, torch.zeros_like(score_p))

        log_score_q = score_q.log()
        score_q = torch.scatter(score_q, -1, x, torch.zeros_like(score_q))

        neg_term = score_p * log_score_q

        # constant factor
        const = score_p * (log_score_p - 1)

        #positive term
        pos_term = score_q

        unscaled_ret_value = pos_term - neg_term + const

        transp_rate = self.transp_rate(x)
        scale_factor = torch.scatter(transp_rate, -1, x[...,None], torch.zeros_like(transp_rate))

        x = x.squeeze(-1)

        transp_rate = self.transp_rate(x)
        try:
            scale_factor = torch.scatter(transp_rate, -1, x[...,None], torch.zeros_like(transp_rate))
        except:
            raise ValueError(f"Could not scatter {transp_rate.shape} with {x.shape}")
        scale_factor = scale_factor * dsigma[..., None]

        ret = scale_factor * unscaled_ret_value
        ret = ret.reshape(ret.shape[0], -1)

        ret = ret.sum(dim=-1)

        return ret


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim):
        self._dim = dim
        Q = torch.ones((dim,dim))
        for i in range(dim):    
            Q[i,i] = 1-dim
        self._Q = Q
        self.Q_seq = None
        self.num_tokens = None

    @property
    def dim(self):
        return self._dim
    
    @dim.setter
    def dim(self, value):
        self._dim = value
    
    @property
    def absorb(self):
        return False
    
    @property
    def Q(self):
        return self._Q

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        try:
            trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
            trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
            trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        except:
            original_shape = i.shape
            i = i.reshape(original_shape[0],-1)
            trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
            trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
            trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
            trans = trans.reshape(*original_shape, self.dim)
        return trans
    
    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        
        # print(f"Examples:\n\tMove chance: {move_chance[:5]}\n\tMove indices: {move_indices[:5]}\n\tI: {i[:5]}\n\tI pert: {i_pert[:5]}")
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)        

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score.exp()
        # print(f"Sexp shape: {sexp.shape}")
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        # print(f"pos_term shape: {pos_term.shape}")
        return pos_term - neg_term + const

    def derivative_score_entropy(self, score, sigma, x, x0):
        
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)        

        # negative term
        score = 1/score.exp()
        score = torch.scatter(score, -1, x[..., None], torch.zeros_like(score))
        try:
            neg_term = ratio[...,None] * score
        except:
            raise UserWarning(f"Could not scatter {score.shape} with {x.shape}")

        neg_term = neg_term.reshape(neg_term.shape[0], -1).sum(dim=-1)
        pos_term = 1
        # print(f"pos_term shape: {pos_term.shape}")
        return pos_term - neg_term
    
    def score_logprobability(self, score_p, dsigma, x, sigma=None):

        x = x.unsqueeze(-1)

        log_score_p = torch.scatter(score_p.log(), -1, x, torch.zeros_like(score_p))
        score_p = torch.scatter(score_p, -1, x, torch.zeros_like(score_p))

        # constant factor
        const = score_p * (log_score_p - 1) + 1

        unscaled_ret_value = const

        x = x.squeeze(-1)

        transp_rate = self.transp_rate(x)
        try:
            scale_factor = torch.scatter(transp_rate, -1, x[...,None], torch.zeros_like(transp_rate))
        except:
            raise ValueError(f"Could not scatter {transp_rate.shape} with {x.shape}")
        scale_factor = scale_factor * dsigma[..., None]

        ret = scale_factor * unscaled_ret_value
        ret = ret.reshape(ret.shape[0], -1).sum(dim=-1)

        return ret


class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
    @dim.setter
    def dim(self, value):
        self._dim = value
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        try:
            move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        except:
            raise ValueError(f"Incompatible shapes: {i.shape} and {sigma.shape}")
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        # raise UserWarning(f"Devices: {esigm1.device}, {x.device}, {x0.device}, {rel_ind.device}, {score.device}")
        # raise UserWarning(f"Dimensions: {score.shape}, {esigm1.shape}, {rel_ind.shape}, {x0.shape}, {x.shape}, rel_ind={rel_ind}")

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
    
    def score_logprobability(self, score_p, dsigma, x, sigma=None):

        assert sigma is not None, "sigma must be provided for absorbing graph"

        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        x = x.unsqueeze(-1)
        
        log_score_p = torch.scatter(score_p.log(), -1, x, torch.zeros_like(score_p))
        score_p = torch.scatter(score_p, -1, x, torch.zeros_like(score_p))

        # raise UserWarning(f"Devices: {esigm1.device}, {x.device}, {x0.device}, {rel_ind.device}, {score.device}")
        # raise UserWarning(f"Dimensions: {score.shape}, {esigm1.shape}, {rel_ind.shape}, {x0.shape}, {x.shape}, rel_ind={rel_ind}")
        esigm1 = esigm1.unsqueeze(-1)
        ratio = 1 / esigm1.expand_as(score_p)
        ratio = ratio/(self.dim-1)

        rel_ind = rel_ind.unsqueeze(-1).expand_as(score_p)

        # constant term
        const = score_p * (log_score_p - 1)
        
        pos_term = ratio
        neg_term = score_p * ratio.log()

        # unscaled_ret_value = pos_term - neg_term + const
        unscaled_ret_value = pos_term - neg_term + const

        x = x.squeeze(-1)

        transp_rate = self.transp_rate(x)
        try:
            scale_factor = torch.scatter(transp_rate, -1, x[...,None], torch.zeros_like(transp_rate))
        except:
            raise ValueError(f"Could not scatter {transp_rate.shape} with {x.shape}")
        scale_factor = scale_factor * dsigma[..., None]
        # scale_factor = dsigma[..., None]

        # raise UserWarning(f"Shapes: {unscaled_ret_value.shape}, {scale_factor.shape}")

        ret = scale_factor * unscaled_ret_value
        ret = torch.where(rel_ind, ret, torch.zeros_like(ret))
        try:
            ret = ret[:,:,:-1]
        except:
            raise UserWarning(f"Could not slice {ret.shape}")

        # raise UserWarning(f"Shape is {ret.shape}, example: {ret[:5]}, x: {x[:5]}") # 256,9,2
        ret = ret.reshape(ret.shape[0], -1)
        ret = ret.sum(dim=-1)
        # raise UserWarning(f"Shape is {ret.shape}") # 256,9

        return ret