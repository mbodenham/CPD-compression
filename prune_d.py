import torch
import distiller

from CPD import CPD, CPD_darknet19

device = torch.device('cuda')
state_dict = torch.load('CPD_darknet19.pth', map_location=torch.device(device))
model = CPD().to(device)

df = distiller.weights_sparsity_summary(model, True)
print(df)

t, total = distiller.weights_sparsity_tbl_summary(model, True)
print(t)
print(total)

#convs = []
#for name, module in model.named_modules():
#	if len(module._modules) == 0:
#		if module.__class__.__name__ == 'Conv2d':
#			convs.append([name, module.__class__.__name__])
#			print(convs[-1])
#
#print(len(convs))
