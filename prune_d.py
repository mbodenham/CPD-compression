import torch
import distiller

from CPD import CPD, CPD_darknet19

device = torch.device('cuda')
state_dict = torch.load('CPD_darknet19.pth', map_location=torch.device(device))
model = CPD().to(device)

 distiller.model_summary(model, 'sparsity')
