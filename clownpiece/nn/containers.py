# Module Sequential / List / Dict
from typing import Iterable
from clownpiece.nn.module import Module


class Sequential(Module):
  def __init__(self, *args: Module):
    super().__init__()
    for i, module in enumerate(args):
      self.add_module(str(i), module)
      
  def __getitem__(self, idx):
    return self._modules[str(idx)]
  
  def __setitem__(self, idx, module):
    self.add_module(str(idx), module)
    
  def __len__(self):
    return len(self._modules)
  
  def forward(self, x):
    for module in self._modules.values():
      x = module(x)
    return x
  

class ModuleList(Module):
  def __init__(self, modules: Iterable[Module] = None):
    super().__init__()
    if modules is not None:
      for i, module in enumerate(modules):
        self.add_module(str(i), module)
        
  def __getitem__(self, idx):
    return self._modules[str(idx)]
  
  def __setitem__(self, idx, module):
    self.add_module(str(idx), module)
    
  def __len__(self):
    return len(self._modules)
  
  def __add__(self, other):
    if isinstance(other, ModuleList):
      return ModuleList(list(self._modules.values()) + list(other._modules.values()))
    elif isinstance(other, Module):
      return ModuleList(list(self._modules.values()) + [other])
    else:
      raise TypeError("Can only add ModuleList or Module to ModuleList")
  
  def append(self, module):
    self.add_module(str(len(self)), module)
    return self
  
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")


class ModuleDict(Module):
  def __init__(self, modules: dict[str, Module] = None):
    super().__init__()
    if modules is not None:
      for k, v in modules.items():
        self.add_module(k, v)
        
  def __getitem__(self, key):
    return self._modules[key]
  
  def __setitem__(self, key, module):
    self.add_module(key, module)
    
  def __len__(self):
    return len(self._modules)
  
  def keys(self):
    return self._modules.keys()
  
  def values(self):
    return self._modules.values()
  
  def items(self):
    return self._modules.items()
  
  def update(self, modules):
    for k, v in modules.items():
      self.add_module(k, v)
      
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")

