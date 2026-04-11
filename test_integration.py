import black_ferox
from black_ferox import black_nn, black_optim
from black_ferox.black_nn.black_transformers import BlackGPT
import numpy as np

def run_test():
    print("1. import black_ferox - OK")
    
    # 2. Build BlackGPT
    model = BlackGPT(
        black_vocab_size=100,
        black_n_embd=32,
        black_n_head=2,
        black_n_layer=2,
        black_block_size=64
    )
    print("2. build BlackGPT - OK")
    
    # x = black_ferox.black_tensor(np.random.randint(0, 100, size=(2, 16)))
    # out = model(x)
    # loss = out.black_sum()
    # loss.black_backward()
    
    # 3. Count parameters
    params = model.black_parameters()
    total_params = sum(np.prod(p.black_data.black_shape()) for p in params)
    print(f"3. count properties: {total_params} - OK")
    
    # 4. Forward pass
    x_data = np.random.randint(0, 100, size=(2, 16)).astype(np.int32)
    x = black_ferox.black_tensor(x_data)
    logits = model(x)
    print(f"4. forward pass - OK")
    
    # 5. Calculate loss
    loss = logits.black_sum()
    print(f"5. calculate loss: {loss.black_data.black_item()} - OK")
    
    # 6. Backward
    loss.black_backward()
    print("6. backward pass - OK")
    
    # 7. Optimizer step
    optimizer = black_optim.BlackAdamW(model.black_parameters(), black_lr=0.01)
    optimizer.black_step()
    print("7. optimizer step - OK")

if __name__ == "__main__":
    run_test()
