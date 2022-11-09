import torch
from torch import Tensor, nn
from torch.nn import Linear, Parameter, ReLU, Sequential
from torch.types import _dtype
import torch.autograd as autograd
from typing import Callable


def anderson_solver(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
        """ 
        Anderson acceleration for fixed point iteration. 
        
        Parameters
        ----------
        ...
        beta : 1.0 indicates no relaxation (or mixing) 
        """
        bsz, d = x0.view(x0.shape[0], -1).shape
        X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
        X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
        
        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1
        
        res = []
        for k in range(2, max_iter):
            n = min(k, m)
            G = F[:,:n]-X[:,:n]
            H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.linalg.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
            
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
            F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
            res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
            if (res[-1] < tol):
                break
        return X[:,k%m].view_as(x0), res



class Mask(nn.Module):
    """A mask non-linearity."""
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor) -> Tensor:
        return ~self.mask*input


class SkipConnection(nn.Module):
    """A skip connection operation."""
    value: Tensor

    def __init__(self, value: Tensor):
        super(SkipConnection, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input + self.value


class NeuMissBlock(nn.Module):
    """The NeuMiss block from "What’s a good imputation to predict with
    missing values?" by Marine Le Morvan, Julie Josse, Erwan Scornet,
    Gael Varoquaux."""

    def __init__(self, n_features: int, depth: int,
                 dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs and outputs of the NeuMiss block.
        depth : int
            Number of layers (Neumann iterations) in the NeuMiss block.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.depth = depth
        self.dtype = dtype
        self.mu = Parameter(torch.empty(n_features, dtype=dtype))
        self.linear = Linear(n_features, n_features, bias=False)  #, dtype=dtype)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)  # Initialize mask non-linearity
        x = torch.nan_to_num(x)  # Fill missing values with 0
        h = x - mask(self.mu)  # Subtract masked parameter mu
        skip = SkipConnection(h)  # Initialize skip connection with this value

        layer = [self.linear, mask, skip]  # One Neumann iteration
        layers = Sequential(*(layer*self.depth))  # Neumann block

        return layers(h)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mu)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def extra_repr(self) -> str:
        return 'depth={}'.format(self.depth)


class NeuMissMLP(nn.Module):
    """A NeuMiss block followed by a MLP."""

    def __init__(self, n_features: int, neumiss_depth: int, mlp_depth: int,
                 mlp_width: int = None, dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs.
        neumiss_depth : int
            Number of layers in the NeuMiss block.
        mlp_depth : int
            Number of hidden layers in the MLP.
        mlp_width : int
            Width of the MLP. If None take mlp_width=n_features. Default: None.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.n_features = n_features
        self.neumiss_depth = neumiss_depth
        self.mlp_depth = mlp_depth
        self.dtype = dtype
        mlp_width = n_features if mlp_width is None else mlp_width
        self.mlp_width = mlp_width

        b = int(mlp_depth >= 1)
        last_layer_width = mlp_width if b else n_features
        self.layers = Sequential(
            NeuMissBlock(n_features, neumiss_depth, dtype),
            *[Linear(n_features, mlp_width, dtype=dtype), ReLU()]*b,
            *[Linear(mlp_width, mlp_width, dtype=dtype), ReLU()]*b*(mlp_depth-1),
            *[Linear(last_layer_width, 1, dtype=dtype)],
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return out.squeeze()


class NeuMissIteration(nn.Module):
    """Single iteration of the NeuMiss block."""

    def __init__(self, n_features: int,
                 dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs and outputs of the NeuMiss block.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.
        """
        super().__init__()
        self.dtype = dtype
        self.mu = Parameter(torch.empty(n_features, dtype=dtype))
        self.linear = Linear(n_features, n_features, bias=False)
        self.reset_parameters()

    def forward(self, x: Tensor, mask: Mask, skip: SkipConnection) -> Tensor:
        return skip(mask(self.linear(x)))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mu)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)


class NeuMissDEQBlock(nn.Module):
    """The NeuMiss block reformulated as a Deep Equilibrium model [Bai et al., 2019]
    instead of an unrolled order-l approximation"""

    def __init__(self, 
                 n_features: int, 
                 solver: Callable = anderson_solver,
                 dtype: _dtype = torch.float,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs and outputs of the NeuMiss block.
        solver : Callable
            Function that solves a fixed-point iteration problem.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.
        """
        super().__init__()
        self.dtype = dtype
        self.neumiss = NeuMissIteration(n_features=n_features, dtype=dtype)
        self.solver = solver
        self.kwargs = kwargs
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)  # Initialize mask non-linearity
        x = torch.nan_to_num(x)  # Fill missing values with 0
        h = x - mask(self.neumiss.mu)  # Subtract masked parameter mu
        skip = SkipConnection(h)  # Initialize skip connection with this value
        
        # compute forward pass outside of autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.neumiss(z, mask, skip), 
                                              h,  # torch.zeros_like(h),
                                              **self.kwargs)
        # re-engage autograd tape at the equilibrium point
        z = self.neumiss(z, mask, skip)
        
        # set up the vector jacobian product
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, mask, skip)
        def backward_hook(in_grad):
            g, self.backward_res = self.solver(lambda g: autograd.grad(f0, z0, g, retain_graph=True)[0] + in_grad,
                                               in_grad, **self.kwargs)  # autograd.grad computes a VJP of df0/dz0 with g
            return g
                
        z.register_hook(backward_hook)
        return z
