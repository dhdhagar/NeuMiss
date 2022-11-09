import torch
from neumiss.NeuMissBlock import NeuMissDEQBlock, NeuMissBlock


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    X = torch.log(1e-4 + torch.randint(-1,5,(5,10)))
    layer = NeuMissDEQBlock(X.shape[1])
    n_params = count_parameters(layer)
    layer2 = NeuMissBlock(X.shape[1], 10)
    n_params2 = count_parameters(layer2)
    
    
    print("X:")
    print(X)
    
    print("layer(X):")
    out = layer(X)
    print(out)
    print("count: ", n_params)

    print("layer2(X):")
    out2 = layer2(X)
    print(out2)
    print("count: ", n_params2)
