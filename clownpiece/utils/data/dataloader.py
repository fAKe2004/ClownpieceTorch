from clownpiece.tensor import stack, Tensor
from clownpiece.utils_ import ceil_div

import random
class DefaultSampler:
  def __init__(self, length, shuffle):
    self.length = length
    self.shuffle = shuffle  
      
  def __iter__(self):
    indices = list(range(len(self)))
    if self.shuffle:
      random.shuffle(indices)
    
    for index in indices:
      yield index
    
  def __len__(self):
    return self.length
  
def default_collate_fn(batch):
  is_tuple = isinstance(batch[0], (tuple, list))
  if not is_tuple:
    batch = [(b,) for b in batch]

  stacked_batch = []
  for i in range(len(batch[0])):
    if not isinstance(batch[0][i], Tensor):
      stacked_batch.append(stack([Tensor(b[i], requires_grad=False) for b in batch]))
    else:
      stacked_batch.append(stack([b[i] for b in batch]))
    
  if not is_tuple:
    stacked_batch = stacked_batch[0]
  return stacked_batch

class Dataloader():
  
  def __init__(self, 
               dataset, 
               batch_size=1, 
               shuffle=False, 
               drop_last=False, 
               sampler=None,
               collate_fn=None,
              ):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last

    assert not shuffle or sampler is None, "Cannot specify both shuffle and sampler"
    
    self.sampler = sampler if sampler is not None else DefaultSampler(len(dataset), shuffle)
    self.collate_fn = collate_fn if collate_fn is not None else default_collate_fn
    
  def __iter__(self):
    batch = []
    for index in self.sampler:
      try:
        batch.append(self.dataset[index])
        if len(batch) == self.batch_size:
          yield self.collate_fn(batch)
          batch = []
      except IndexError:
        raise RuntimeError(f"Dataloader: Index {index} out of range for dataset of length {len(self.dataset)}")
      
    if not self.drop_last and len(batch) > 0:
      yield self.collate_fn(batch)
          
  def __len__(self):
    if self.drop_last:
      return len(self.sampler) // self.batch_size
    else:
      return ceil_div(len(self.sampler), self.batch_size)