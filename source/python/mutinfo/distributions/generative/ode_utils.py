from scipy import integrate
import torch
import torchdiffeq

def get_ode_sampler(drift_fn, inverse_scaler,
                    rtol=1e-5, atol=1e-5, step_size=1e-2,
                    method='rk4', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    drift_fn: a callable which produces a drift given model, time and samples.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples.
  """

  def ode_sampler(x):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      x: initial sample from the prior distribution.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():

      def ode_func(t, x):
        x = torch.tensor(x, device=device, dtype=torch.float32)
        vec_t = torch.ones(x.shape[0], device=x.device) * t
        drift = drift_fn(x, vec_t)
        return drift

      # Black-box ODE solver for the probability flow ODE using torchdiffeq
      t = torch.tensor([1.0, eps]).to(device, dtype=torch.float32)
      with torch.no_grad():
        solution = torchdiffeq.odeint(ode_func, x, t, rtol=rtol, atol=atol, method=method, options={'step_size':step_size})
      # TODO: I hard coded T=1.0 but it would be better to get it somewhere
      x = solution[-1].to(device)

      x = inverse_scaler(x)
      return x

  return ode_sampler