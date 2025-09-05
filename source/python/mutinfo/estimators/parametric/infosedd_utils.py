import torch
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np

from itertools import cycle

def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """

        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma)

    return model_fn

def get_score_fn(model, train=False, sampling=False):
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        def score_fn(x, sigma):
            
            score = model_fn(x, sigma)
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn

def get_loss_fn(noise, graph, train, is_parametric_marginal, variant, sampling_eps=1e-3):

    def loss_fn(model, x, y):
        """
        Batch shape: [B, L] int. D given from graph
        """
        x_indices = list(range(x.shape[1]))
        y_indices = list(range(x.shape[1], x.shape[1] + y.shape[1]))
        batch = torch.cat([x, y], dim=-1)
        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        
        sigma, dsigma = noise(t)

        if is_parametric_marginal and variant == "j":
            marginal_flag = np.random.randint(0, 3)
            if marginal_flag == 0:
                absorb_indices = x_indices
            elif marginal_flag == 1:
                absorb_indices = y_indices
            else:
                absorb_indices = None
        elif is_parametric_marginal and variant == "c":
            marginal_flag = np.random.randint(0, 2)
            absorb_indices = x_indices
        else:
            absorb_indices = None
            marginal_flag = None
        
        perturbed_batch = graph.sample_transition(batch, sigma[:, None])
        if is_parametric_marginal and variant == "j":
            if absorb_indices is not None:
                perturbed_batch[:, absorb_indices] = graph.dim - 1
        elif is_parametric_marginal and variant == "c":
            if marginal_flag == 0:
                perturbed_batch[:, absorb_indices] = batch[:, absorb_indices]
            elif marginal_flag == 1:
                perturbed_batch[:, absorb_indices] = graph.dim - 1
            else:
                raise ValueError(f"Invalid marginal_flag: {marginal_flag}")

        log_score_fn = get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)

        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        if is_parametric_marginal:
            if variant == "j":
                if marginal_flag == 0:
                    loss = loss[:, y_indices]
                elif marginal_flag == 1:
                    loss = loss[:, x_indices]
            elif variant == "c":
                loss = loss[:, y_indices]

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss.mean()
    
    return loss_fn

def get_mutinfo_step_fn(graph, noise, variant, proj_fn = lambda x, is_score: x):

    def mutinfo_step_fn(model, x, y, return_noise=False):
        x_indices = list(range(x.shape[1]))
        y_indices = list(range(x.shape[1], x.shape[1] + y.shape[1]))
        batch = torch.cat([x, y], dim=-1)
        score_fn = get_score_fn(model, train=False, sampling=True)
        with torch.no_grad():
            t = torch.rand(batch.shape[0], 1).to(batch.device)
            sigma, dsigma = noise(t)
            
            perturbed_batch = graph.sample_transition(batch, sigma)
            perturbed_batch = proj_fn(perturbed_batch, is_score=False)

            if variant == "j":

                perturbed_batch_x = perturbed_batch.clone()
                perturbed_batch_x[:, y_indices] = graph.dim - 1

                perturbed_batch_y = perturbed_batch.clone()
                perturbed_batch_y[:, x_indices] = graph.dim - 1

                score_joint = score_fn(perturbed_batch, sigma)
                score_marginal_x = score_fn(perturbed_batch_x, sigma)
                score_marginal_y = score_fn(perturbed_batch_y, sigma)

                score_marginal_x = score_marginal_x[:, x_indices]

                score_marginal_y = score_marginal_y[:, y_indices]

                score_marginal = torch.cat([score_marginal_x, score_marginal_y], dim=1)

                score_marginal = proj_fn(score_marginal, is_score=True)
                score_joint = proj_fn(score_joint, is_score=True)

                # raise UserWarning(f"Score joint examples {score_joint[:5]}, x examples {perturbed_batch[:5]}")
                perturbed_batch = proj_fn(perturbed_batch, is_score=True)
                divergence_estimate = graph.score_divergence(score_joint, score_marginal, dsigma, perturbed_batch)
            elif variant == "c":
                perturbed_batch_marginal = perturbed_batch.clone()
                perturbed_batch_marginal[:, x_indices] = graph.dim - 1
                perturbed_batch_conditional = perturbed_batch.clone()
                perturbed_batch_conditional[:, x_indices] = batch[:, x_indices]
                
                score_marginal = score_fn(perturbed_batch, sigma)
                score_conditional = score_fn(perturbed_batch, sigma)

                score_marginal = proj_fn(score_marginal, is_score=True)[:, y_indices]
                score_conditional = proj_fn(score_conditional, is_score=True)[:, y_indices]

                perturbed_batch = perturbed_batch[:, y_indices]

                divergence_estimate = graph.score_divergence(score_conditional, score_marginal, dsigma, perturbed_batch)
            
            return divergence_estimate.mean().item()
    
    return mutinfo_step_fn