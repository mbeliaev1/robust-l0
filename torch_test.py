import torch

print(torch.cuda.get_arch_list())
print(torch.version.cuda)

t = torch.zeros(3)
print(t)