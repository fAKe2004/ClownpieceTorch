from typing import Dict, Iterable, List, Optional, Union, Any

from clownpiece.tensor import Tensor, ones_like, zeros_like
from clownpiece.utils import wrap_tuple

"""
    Autograd Module
"""

class Node():
    node_id: int
    topological_nr: int
    next_edges: List["Edge"]
    def __init__(self):
        self.node_id = None
        self.topological_nr = None
        self.next_edges = []
        
    def run(self, *args, **kargs):
        raise NotImplementedError("run method not implemented for abstract Node instance")
    
    # define __hash__ and __eq__ to use Node as dict's key
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id
    
class Edge():
  
    input_nr: int
    node: Optional[Node]

    def __init__(self, input_nr: int, node: Optional[Node]):
        self.input_nr = input_nr
        self.node = node
    
    @staticmethod
    def gradient_edge(tensor: Tensor) -> "Edge":
        from clownpiece.autograd.function import AccumulateGrad
        if not isinstance(tensor, Tensor):
            return Edge(input_nr = 0, node = None)
        if tensor.requires_grad:
            if tensor.grad_fn is None:
                return Edge(input_nr = 0, node = AccumulateGrad(tensor))
            else:
                return Edge(input_nr = tensor.output_nr, node = tensor.grad_fn)
        else:
            return Edge(input_nr = 0, node = None)

class GraphRoot(Node):
    """
    Root node in the computation graph.
    """
    
    def __init__(self, tensor: Tensor, grad: Tensor):
        super().__init__()
        self.tensor = tensor
        self.grad = grad
        self.next_edges = [Edge.gradient_edge(tensor)]
    
    def run(self, *args, **kargs):
        return self.grad

class NodeTask():
    """
    NodeTask wraps a Node and all its input. 
    It's a ready-to-run Node in GraphTask.
    """
    base: "GraphTask"
    node: Node
    inputs: List[Tensor]
    
    def __init__(self, node: Node, inputs: List[Tensor], base: "GraphTask"):
        self.base = base
        self.node = node
        self.inputs = inputs
        
    def run(self):
        outputs = self.node.run(*self.inputs)
        outputs = wrap_tuple(outputs)
        for output, edge in zip(outputs, self.node.next_edges):
            self.base.fill_input(edge.node, output, edge.input_nr)
        

class GraphTask():
    
    """
    GraphTask wraps the execution of a computation graph.
    """
    
    roots: List[Node]
    nodes: List[Node]
    dependencies: Dict[Node, int]
    inputs_buffer: Dict[Node, List[Tensor]]
    
    def __init__(self, roots: List[Node]):
        roots = wrap_tuple(roots)
        roots = [root for root in roots if root is not None]
        
        if not roots:
            raise ValueError("roots is empty")
    
        self.roots = roots
        self.nodes = []
        self.dependencies = {}
        self.inputs_buffer = {}
        self._construct_graph()
        
    def _construct_graph(self):
        """
        Constructs the graph from the root node
        """        
        self.inputs_buffer = {}
        
        node_id_cnt = 0
        def _assign_node_id(node: Node):
            nonlocal node_id_cnt
            if node is None or node.node_id is not None:
                return
            self.nodes.append(node)
            
            node.node_id = node_id_cnt
            node_id_cnt += 1
            for edge in node.next_edges:
                _assign_node_id(edge.node)
        
        for root in self.roots:
            _assign_node_id(root)
        
        self.dependencies = {
            node: 0 for node in self.nodes
        }
        
        num_inputs = {
            node: 1 for node in self.nodes
        }
        
        # build dependencies
        for node in self.nodes:
            for edge in node.next_edges:
                if edge.node is not None:
                    self.dependencies[edge.node] += 1
                    num_inputs[edge.node] = max(num_inputs[edge.node], edge.input_nr + 1)

        # build inputs buffer
        self.inputs_buffer = {
            node: [None] * num_inputs[node] for node in self.nodes
        }
        
    def run(self):
        self._run_single_thread()
        # self._run_multi_thread()
        
    def _run_single_thread(self):
        ready_queue: List[NodeTask] = [
            NodeTask(root, (), self) for root in self.roots
        ]
        
        # topological sort
        while ready_queue:
            node_task = ready_queue.pop(0)
            node = node_task.node
            
            if node is None:
                continue
            
            node_task.run()
                        
            for edge in node.next_edges:
                if edge.node is not None:
                    self.dependencies[edge.node] -= 1
                    
                    if self.dependencies[edge.node] == 0:
                        ready_queue += [NodeTask(edge.node, self.inputs_buffer[edge.node], self)]
                        del self.inputs_buffer[edge.node]
                        
                        
    def _run_multi_thread(self):
        import threading
        ready_queue: List[NodeTask] = [
            NodeTask(root, (), self) for root in self.roots
        ]
        completed = 0
        
        exceptions = [] 
                
        NUM_THREAD = 4
        
        def worker():
            nonlocal ready_queue, exceptions, completed
            try:
                while completed < len(self.nodes):
                    while ready_queue:
                        node_task = ready_queue.pop(0)
                        node = node_task.node
                        if node is None:
                            continue
                        node_task.run()
                        for edge in node.next_edges:
                            if edge.node is not None:
                                self.dependencies[edge.node] -= 1
                                if self.dependencies[edge.node] == 0:
                                    ready_queue.append(NodeTask(edge.node, self.inputs_buffer[edge.node], self))
                                    del self.inputs_buffer[edge.node]
                        completed += 1
            except Exception as exc:
                exceptions.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(NUM_THREAD)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
            
        if exceptions:
            raise Exception(f"Exceptions occurred in threads: {exceptions}")
                    
    def fill_input(self, node: Node, input_grad: Tensor, input_nr: int):
        """
        Set inputs buffer for a node
        """
        if node is None:
            return
        
        inputs = self.inputs_buffer[node]
        assert 0 <= input_nr < len(inputs), "Input number out of range"
                
        if input_grad is None:
            return
        
        if inputs[input_nr] is None:
            inputs[input_nr] = zeros_like(input_grad)

        inputs[input_nr] += input_grad


"""
    Execute backward pass.    
"""
def backward(tensors: Union[Tensor, List[Tensor]], grads: Optional[Union[Tensor, List[Tensor]]] = None):
    tensors = wrap_tuple(tensors)

    if grads is None:
        grads = [ones_like(tensor) for tensor in tensors]
    grads = wrap_tuple(grads)
    
    # wrap with GraphRoots
    graph_roots = [
        GraphRoot(tensor, grad) for tensor, grad in zip(tensors, grads) if tensor.requires_grad
    ]

    # execute with GraphTask
    gt = GraphTask(graph_roots)
    gt.run()