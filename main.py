import torch
from transformer import Encoder

if __name__ == "__main__":
    model = Encoder(1000, 256, 10, 8, 4, 0.1, 50)
    input = torch.randint(0, 1, (1000, 10))
    output = model(input)
    print(f"input shape: {input.shape}")
    print(f"output shape: {output.shape}")
