from typing import Dict, Iterable, Tuple, Union, Optional
from clownpiece import Tensor, zeros_like
from clownpiece.tensor import empty_like


class Parameter(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=True)
    

class Buffer(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=False)


class Module(object):
  training: bool
  _init_called: bool = False
  
  _parameters: dict[str, Parameter] = {}
  _buffers: dict[str, Tensor] = {}
  _modules: dict[str, 'Module'] = {}
  
  def __init__(self):
    self._init_called = True
    self.training = True
    
    self._parameters = {}
    self._buffers = {}
    self._modules = {}
    
  def train(self, flag: bool = True):
    self.training = flag
    for module in self._modules.values():
      module.train(flag)
    return self

  def eval(self):
    self.train(False)    

  def __setattr__(self, name: str, value):
    if name != "_init_called" and not self._init_called:
      raise RuntimeError(f"Module {self.__class__.__name__} is not initialized. "
                         f"Please call the Module.__init__.")
    
    if name in ["_parameters", "_buffers", "_modules"]:
      return super().__setattr__(name, value)
    
    # remove first
    if name in self._parameters:
      del self._parameters[name]
    if name in self._buffers:
      del self._buffers[name]
    if name in self._modules:
      del self._modules[name]
    
    # add 
    if isinstance(value, Parameter):
      self._parameters[name] = value
    elif isinstance(value, Tensor):
      self._buffers[name] = value
    elif isinstance(value, Module):
      self._modules[name] = value
    else:
      super().__setattr__(name, value)
      
  def __getattr__(self, name: str):
    if name in self._parameters:
      return self._parameters[name]
    elif name in self._buffers:
      return self._buffers[name]
    elif name in self._modules:
      return self._modules[name]
    else:
      raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
  """
    Forward
  """
    
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  """
    Parameters
  """
  def register_parameter(self, name: str, param: Optional[Parameter]):
    if not isinstance(param, Parameter) and not param is None:
      raise TypeError(f"Expected Parameter or None, got {type(param)}")
    self._parameters[name] = param

  def parameters(self, recursive: bool = True) -> Iterable[Parameter]:
    for param in self._parameters.values():
      yield param
    if recursive:
      for module in self._modules.values():
        yield from module.parameters()

  def named_parameters(self, recursive: bool = True) -> Iterable[Parameter]:
    return self._named_parameters("", recursive=recursive)
  
  def _named_parameters(self, prefix: str, recursive: bool = True) -> Iterable[Tuple[str, Buffer]]:
    for name, param in self._parameters.items():
      yield prefix + name, param
    if recursive:
      for name, module in self._modules.items():
        yield from module._named_parameters(prefix + name + ".")

  """
    Buffers
  """
  def register_buffer(self, name: str, buffer: Optional[Buffer]):
    if not isinstance(buffer, Buffer) and not buffer is None:
      raise TypeError(f"Expected Buffer or None, got {type(buffer)}")
    self._buffers[name] = buffer

  def buffers(self, recursive=True) -> Iterable[Buffer]:
    for buffer in self._buffers.values():
      yield buffer
    if recursive:
      for module in self._modules.values():
        yield from module.buffers()
  
  def named_buffers(self, recursive: bool =True) -> Iterable[Tuple[str, Buffer]]:
    return self._named_buffers("", recursive=recursive)
  
  def _named_buffers(self, prefix: str, recursive: bool = True) -> Iterable[Tuple[str, Buffer]]:
    for name, buffer in self._buffers.items():
      yield prefix + name, buffer
    if recursive:
      for name, module in self._modules.items():
        yield from module._named_buffers(prefix + name + ".")
  
  """
    Modules
  """
  
  def register_module(self, name: str, module: 'Module'):
    if not isinstance(module, Module):
      raise TypeError(f"Expected Module, got {type(module)}")
    self._modules[name] = module
    
  def modules(self, recursive: bool = True) -> Iterable['Module']:
    for module in self._modules.values():
      yield module
    if recursive:
      for module in self._modules.values():
        yield from module.modules()
  
  def named_modules(self, recursive: bool = True) -> Iterable[Tuple[str, 'Module']]:
    return self._named_modules(self.__class__.__name__, recursive=recursive)
  
  def _named_modules(self, prefix: str, recursive: bool = True) -> Iterable[Tuple[str, 'Module']]:
    for name, module in self._modules.items():
      yield prefix + name, module
    if recursive:
      for name, module in self._modules.items():
        yield from module._named_modules(prefix + name + ".")
    
  """
    State Dict
  """

  def state_dict(self) -> Dict:
    return self._state_dict("")
  
  def _state_dict(self, prefix: str) -> Dict[str, Tensor]:
    state = {}
    for name, param in self._parameters.items():
      state[prefix + name] = param
    for name, buffer in self._buffers.items():
      state[prefix + name] = buffer
    for name, module in self._modules.items():
      state.update(module._state_dict(prefix + name + "."))
    return state
    
  def load_state_dict(self, state: Dict[str, Tensor], strict: bool = True):
    self._load_state_dict(self, state=state, strict=strict)
    
  def _load_state_dict(self, state: Dict, strict: bool) -> int:
        
    def check_shape_match(name: str, dst: Tensor, src: Tensor):
      if dst is not None != src is not None:
        raise RuntimeError(f"load_state_dict: type mismatch for {name}: "
                           f"expected {'Tensor' if dst is not None else 'None'}, "
                           f"got {'Tensor' if src is not None else 'None'}")
      if dst is None:
        return
      if param.shape != state.shape:
        raise RuntimeError(f"load_state_dict: shape mismatch for {name}: "
                           f"expected {param.shape}, got {state.shape}")
    
    for name, param in self._parameters.items():
      if name in state:
        if strict:
          check_shape_match(name, param, state[name])
        if param is not None:
          param.copy_(state[name])
        del state[name]
      elif strict:
        raise RuntimeError(f"load_state_dict: missing key {name}") 
      
    for name, buffer in self._buffers.items():
      if name in state:
        if strict:
          check_shape_match(name, buffer, state[name])
        if buffer is not None:
          buffer.copy_(state[name])
        del state[name]
        
      elif strict:
        raise RuntimeError(f"load_state_dict: missing key {name}")
      
    for name, module in self._modules.items():
      sub_state = {
        k: v for k, v in state.items() if k.startswith(name + ".")
      }
      module._load_state_dict(sub_state)

      state = {
        k: v for k, v in state.items() if not k.startswith(name + ".")
      }
      
    if strict and len(state) != 0:
      raise RuntimeError(f"load_state_dict: unused key {list(state.keys())}")
    
    
  """
    Printing
  """
  def __repr__(self):
    extra = self.extra_repr()
    child_lines = []

    for name, module in self._modules.items():
        mod_str = repr(module)
        mod_str = Module._addindent(mod_str, 2)
        child_lines.append(f'({name}): {mod_str}')

    lines = []
    if extra:
        lines.append(extra)
    lines.extend(child_lines)

    return self.__class__.__name__ + '(' + ('\n  ' + '\n  '.join(lines) + '\n)' if lines else ')')

  @staticmethod
  def _addindent(s: str, num_spaces: int) -> str:
    indent = ' ' * num_spaces
    return indent + s.replace('\n', '\n' + indent)
    
  def extra_repr(self) -> str:
    return ""