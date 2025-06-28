from typing import Callable, List, Any
import os
from PIL import Image
import numpy as np

class Dataset():
  
  def __init__(self):
    pass
  
  def __getitem__(self, index):
    """
    Returns the item at the given index.
    """
    raise NotImplementedError("Dataset __getitem__ method not implemented")
  
  def __len__(self):
    raise NotImplementedError("Dataset __len__ method not implemented")
  
"""
CSV
"""
  
class CSVDataset(Dataset):
  """
  A dataset that reads data from a CSV file.
  """
  file_path: str
  data: List[Any]
  transform: Callable
  
  def __init__(self, file_path: str, transform: Callable = None):
    super().__init__()
    self.file_path = file_path
    self.data = []
    if transform is None:
      transform = lambda x: x
    self.transform = transform
    self.load_data()
  
  def load_data(self):
    import csv
    with open(self.file_path, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        self.data.append(self.transform(row))
  
  def __getitem__(self, index):
    return self.data[index]
  
  def __len__(self):
    return len(self.data)

"""
Image
"""

class ImageDataset(Dataset):
  """
  A dataset that reads images from a directory with subdirectories.
  Subdirectory name is used as label.
  """
  
  data: List[np.ndarray]
  labels: List[int]
  transform: Callable
  class_to_idx: dict[str, int]
  
  def __init__(self, file_path: str, transform: Callable = None):
    super().__init__()
    self.file_path = file_path
    self.data = []
    self.labels = []
    if transform is None:
      transform = lambda x: x
    self.transform = transform
    self.class_to_idx = {}
    self.load_data()
    
  def load_data(self):
    self.class_to_idx = {}
    for class_name in os.listdir(self.file_path):
      class_path = os.path.join(self.file_path, class_name)
      if os.path.isdir(class_path):
        self.class_to_idx[class_name] = len(self.class_to_idx)
        
        for file_name in os.listdir(class_path):
          file_path = os.path.join(class_path, file_name)
          if os.path.isfile(file_path):
            try:
              # Convert image to numpy array immediately
              image_np = np.array(Image.open(file_path).convert('RGB'))
              # apply transform
              transformed_image = self.transform(image_np)
              self.data.append(transformed_image)
              self.labels.append(self.class_to_idx[class_name])
            except Exception as e:
              print(f"Error loading image {file_path}: {e}")
              
  def __getitem__(self, index):
    return self.data[index], self.labels[index]
  
  def __len__(self):
    return len(self.data)
  
"""
Image Transforms
"""

def sequential_transform(*trans):
  def func(x):
    for t in trans:
      x = t(x)
    return x
  return func

def resize_transform(size):
  def func(x: np.ndarray):
    pil_img = Image.fromarray(x)
    resized_pil = pil_img.resize(size, Image.BILINEAR)
    return np.array(resized_pil)
  return func

def normalize_transform(mean, std):
  def func(x: np.ndarray):
    return (x - mean) / std
  return func
  
def to_tensor_transform():
  from clownpiece import Tensor
  def func(x: np.ndarray):
    return Tensor(x.tolist())
  return func