import torch
import distiller
import pandas as pd

import CPD

device = torch.device('cppu')
#state_dict = torch.load('training/CPD_darknet19_A_pruned/ckpts/CPD_darknet19_A_pruned.pth', map_location=torch.device(device))

model = CPD.load_model('CPD_D19_A').to(device)
#model.load_state_dict(state_dict)

#df, _ = distiller.weights_sparsity_summary(model, True)
#df.to_csv('CPD_darknet19_A_minimal.sparsity.csv')

performance_df = distiller.model_performance_summary(model, torch.rand([1, 3, 352, 352]), 1)
performance_df.to_csv('CPD_darknet19_A.performance.csv')

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
