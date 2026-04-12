# Dynamic shape version of mul.py
# Right now this is a work in progress — setting dynamic=True lets Dynamo
# trace with symbolic shapes, but the SDSC we generate still ends up fully
# concrete. Getting symbolic shapes all the way into the SDSC is tracked
# in issues #220, #1371, #1372, #1373.

import torch

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

def mul_fn(a, b):
    return torch.mul(a, b)

# Compile with dynamic=True 
compiled_fn = torch.compile(mul_fn, dynamic=True)

x = torch.rand(128, 64, dtype=torch.float16)
y = torch.rand(128, 64, dtype=torch.float16)


cpu_result = mul_fn(x, y)

x_device = x.to(DEVICE)
y_device = y.to(DEVICE)
compiled_result = compiled_fn(x_device, y_device).cpu()

# Compare results
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
