# Clownpiece-torch Week 4: Bringing It All Together

Congratulations on making it to the final week! This week, we'll add the final pieces to make `clownpiece-torch` a more complete deep learning library. We'll focus on data loading, optimization, and learning rate scheduling. This will make training models much more convenient and powerful.

---

## Roadmap for Week 4

Our goal is to implement the following key components, which are fundamental to modern deep learning workflows:

1.  **`Dataset` and `DataLoader`**: To handle data loading and batching efficiently.
2.  **`Optimizers`**: To automate the model's weight update process.
3.  **`Learning Rate Schedulers`**: To dynamically adjust the learning rate during training.

---

### 1. Dataset and DataLoader

Currently, we manually load and process our data. Let's create a more structured and reusable way to handle datasets.

#### `Dataset` Class

**Goal:** Create an abstract base `Dataset` class. This will be the foundation for all future datasets.

**Implementation Suggestion:**
-   Create a new file, for example, `clownpiece/utils/data.py`.
-   Define a `Dataset` class with two main methods to be overridden by subclasses:
    -   `__len__(self)`: Should return the total number of samples in the dataset.
    -   `__getitem__(self, idx)`: Should return the sample and its corresponding label at a given index `idx`.

#### `DataLoader` Class

**Goal:** Create a `DataLoader` to automatically batch and shuffle the data from a `Dataset`.

**Implementation Suggestion:**
-   In the same `clownpiece/utils/data.py` file, create a `DataLoader` class.
-   The constructor `__init__` should accept a `Dataset` object, a `batch_size`, and a boolean `shuffle` flag.
-   Make it an iterator by implementing `__iter__` and `__next__`.
-   In `__iter__`, you should generate a list of indices from `0` to `len(dataset) - 1`. If `shuffle=True`, shuffle these indices.
-   `__next__` will take a "batch" of indices, use them to fetch data from the dataset using `__getitem__`, and collate them into tensors for your model.
-   **Keep it simple:** Features like `pin_memory` are great but complex. Focusing on batching and shuffling is perfect for this project.

---

### 2. Optimizers

Let's abstract away the manual weight update loop. This is a huge step towards making your library feel like a professional tool.

#### `Optimizer` Base Class

**Goal:** Create a base class for all optimizers to inherit from.

**Implementation Suggestion:**
-   Create a new directory `clownpiece/optim` with an `__init__.py`.
-   Inside, create a file for your optimizers, e.g., `optimizer.py`.
-   The `Optimizer` base class constructor `__init__` should take the model's `parameters()` and a learning rate `lr`.
-   It needs two methods:
    -   `step()`: This will be implemented by each specific optimizer to update the parameters.
    -   `zero_grad()`: This should iterate through the parameters and set their gradients to `None`.

#### `SGD` Optimizer

**Goal:** Implement the classic Stochastic Gradient Descent optimizer.

**Implementation Suggestion:**
-   Create an `SGD` class that inherits from `Optimizer`.
-   Implement the `step()` method. The update rule is straightforward: `param.data -= self.lr * param.grad` for each parameter.

#### `Adam` Optimizer (Challenge)

**Goal:** Implement the popular Adam optimizer.

**Implementation Suggestion:**
-   Create an `Adam` class that inherits from `Optimizer`.
-   The constructor will need `lr`, `betas=(0.9, 0.999)`, and `eps=1e-8`.
-   You'll need to store the moving averages of the gradient (`m`) and the squared gradient (`v`) for each parameter. You can use lists of tensors initialized to zero for this.
-   Implement the `step()` method according to the Adam algorithm. This is a fantastic challenge and a very practical optimizer to have.

---

### 3. Learning Rate Schedulers

**Goal:** Implement schedulers to adjust the learning rate on the fly, which can significantly improve training.

#### `_LRScheduler` Base Class

**Goal:** A base class for all schedulers.

**Implementation Suggestion:**
-   Create a `lr_scheduler.py` file in `clownpiece/optim`.
-   The `_LRScheduler` class `__init__` should take an `optimizer`.
-   It should have a `step()` method that will be called once per epoch to update the learning rate in the optimizer.

#### `StepLR` and `ExponentialLR`

**Goal:** Implement two common schedulers.

**Implementation Suggestion:**
-   **`StepLR`**: Decays the learning rate by a factor of `gamma` every `step_size` epochs. Its `step()` method will check if the current epoch is a multiple of `step_size`.
-   **`ExponentialLR`**: Decays the learning rate by `gamma` every single epoch. Its `step()` method will multiply the optimizer's current learning rate by `gamma`.

---

This roadmap provides a clear path to round out your project. Finishing these features will give you a very impressive and functional deep learning library. Good luck with the final week!