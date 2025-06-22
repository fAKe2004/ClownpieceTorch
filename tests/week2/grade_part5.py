"""
    Max, Sum, Softmax
"""
import torch
from graderlib import set_debug_mode, testcase, grader_summary, print_separate_line

# Test data
test_data_basic = [[1.0, 2.0, 3.0], [6.0, 4.0, 5.0]]  # For sum and max
test_data_softmax1 = [[1.0, 2.0, 3.0], [0.5, -10, 3.0]]  # For softmax
test_data_softmax2 = [[0.5, 0.1, 0.4], [0., -0.5, 2.0], [1.0, 0.5, 3.0]]  # For softmax

@testcase(name="max_op", score=10, timeout=1000)
def max_op(impl=torch):
    a = impl.Tensor(test_data_basic)
    a.requires_grad_()

    b, _ = a.max(dim=1, keepdims=True)
    b.backward(impl.ones_like(b))
    
    c, _ = a.max(dim=0, keepdims=False)
    c.backward(impl.ones_like(c))

    return a.grad, b, c

@testcase(name="sum_op", score=10)
def sum_op(impl=torch):
    a = impl.Tensor(test_data_basic)
    a.requires_grad_()

    b = a.sum(dim=1, keepdims=True)
    b.backward(impl.ones_like(b))
    
    c = a.sum(dim=0, keepdims=False)
    c.backward(impl.ones_like(c))
    
    d = a.sum(dim=None, keepdims=False)
    d.backward(impl.ones_like(d))
    
    return a.grad, b, c, d

@testcase(name="softmax_op", score=10)
def softmax_op(impl=torch):
    a = impl.Tensor(test_data_softmax1)
    a.requires_grad_()
    b = a.softmax(dim=1)
    b.backward(impl.ones_like(b))
    
    c = impl.Tensor(test_data_softmax2)
    c.requires_grad_()
    d = c.softmax(dim=1)
    d.backward(impl.ones_like(d))
    
    return a.grad, b, c.grad, d

def testsets_part5():
    print_separate_line()
    print("Testing Part5 Max, Sum, Softmax...")
    set_debug_mode(True)
    max_op()
    sum_op()
    softmax_op()

if __name__ == "__main__":
    testsets_part5()
    grader_summary("Part5")