import torch
import distiller
import pandas as pd

from CPD import CPD, CPD_darknet19

device = torch.device('cuda')
state_dict = torch.load('CPD_darknet19.pth', map_location=torch.device(device))
model.load_state_dict(state_dict)
model = CPD_darknet19().to(device)

df = distiller.weights_sparsity_summary(model, True)
df.to_csv('CPD_darknet19.sparsity.csv')

t, total, _ = distiller.weights_sparsity_tbl_summary(model, True)
print(t)
print('Global sparsity:', total)

#convs = []
#for name, module in model.named_modules():
#	if len(module._modules) == 0:
#		if module.__class__.__name__ == 'Conv2d':
#			convs.append([name, module.__class__.__name__])
#			print(convs[-1])
#
#print(len(convs))
