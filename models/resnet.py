import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

####PARAMETERS####
ind_conv = [1,2,3,4,5,6,7]
p_zeros = 0.7 #from 0 to 1

variable_p_zero = False #if set to True then variable p_zero over layers from 1 to 7 is performed
p_zeros_array = [0.7,0.51,0.37,0.26,0.19,0.14,0.1] #evaluated using p_zero(layer_position)=0.9682*EXP(-0.324*layer_position)
# in order to work previous array dimension == dimension of ind_conv
############################################################################################
class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        logits = self.resnet(x)
        return logits

#############################################################################################
class ASH_ResNet(nn.Module):
    def __init__(self):
        super(ASH_ResNet, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.handles_hooks = [] 

    def register_mod_hooks(self):
        nc=0
        conv_count=0
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                nc+=1
                if nc in ind_conv and nc not in [8,13,18]:
                    handle = module.register_forward_hook(self.act_hook(conv_count))
                    self.handles_hooks.append(handle)
                    conv_count+=1

    def act_hook(self,conv_count):
        def hook(module, input, output):
        
            A = output
            d_a = A.device
            p_ones = 1 - p_zeros

            if variable_p_zero == False:
                M = torch.tensor(np.random.choice([0, 1], A.shape, p=[p_zeros, p_ones]), device=d_a)
            else:
                M =  torch.tensor(np.random.choice([0, 1], A.shape, p=[p_zeros_array[conv_count], 1-p_zeros_array[conv_count]]), device=d_a)

            A_binarized = (A > 0).float()
            act_mod= A_binarized * M.float()
            return act_mod
        return hook
    

    def remove_hooks(self):
        for handle in self.handles_hooks:
            handle.remove()
        self.handles_hooks = []
    
    # ## Standard FORWARD ##
    # def forward(self, x):
    #     logits = self.resnet(x)
    #     return logits
    
    ## FORWARD with activation function ##
    def forward(self,x):
        self.register_mod_hooks()
        logits = self.resnet(x)
        self.remove_hooks()       
        return logits


#############################################################################################
class DA_Resnet(ASH_ResNet):
    def __init__(self):
        super(DA_Resnet, self).__init__()
        self.Mt = []

######SAVE ACTIVATIONS######    
    def get_activations(self,ind_list):
        def save_hook(module, input, output):
                self.Mt.insert(ind_list, output.detach())
        return save_hook
                
    ## Standard FORWARD ##
    def forward(self, x):
        logits = self.resnet(x)
        return logits
        
    def register_save_hooks(self):        
        nc=0
        ind_list=0
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                nc = nc+1
                if nc in ind_conv and nc not in [8,13,18]:  
                    handle = module.register_forward_hook(self.get_activations(ind_list))
                    self.handles_hooks.append(handle)
                    ind_list+=1

    ## FORWARD for saving maps ##
    def forward_save_Mt(self,x):
        self.Mt.clear() 
        self.register_save_hooks()
        _ = self.resnet(x)
        self.remove_hooks()
        return self.Mt

##### APPLY ACTIVATION #####
    def apply_hook (self, ind_list, ActMap):
        def act_hook_DA(module, input, output):
            
            A = output
            d_a = A.device
            act_mod= A*ActMap[ind_list]
            
            return act_mod
        return act_hook_DA


    def register_mod_hooks(self,ActMap):
        nc=0
        ind_list=0
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                nc = nc+1
                if nc in ind_conv and nc not in [8,13,18]:
                    handle = module.register_forward_hook(self.apply_hook(ind_list,ActMap))
                    self.handles_hooks.append(handle)
                    ind_list+=1

    ## FORWARD for Domain Adaptation ##
    def forward_DA(self, x, ActMap):
        self.register_mod_hooks(ActMap)
        zs = self.resnet(x)
        self.remove_hooks()
        return zs
    
    


