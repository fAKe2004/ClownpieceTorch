import sys
import os


import torch
import clownpiece
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary
clownpiece.Tensor = clownpiece.TensorBase

@testcase(name="sum1: sum along dim=0", score=10)
def sum1(impl=torch):
    a = impl.Tensor([[1, 2, 3], [4, 5, 6]])
    return a.sum(dim = 0)

@testcase(name="sum2: sum along dim=1", score=10)
def sum2(impl=torch):
    a = impl.Tensor([[1, 2, 3], [4, 5, 6]])
    return a.sum(dim = 1)

@testcase(name="sum3: sum with keepdim", score=10)
def sum3(impl=torch):
    a = impl.Tensor([[1, 2, 3], [4, 5, 6]])
    return a.sum(dim = 1, keepdim = True)

@testcase(name="sum4: sum for higher dimension", score=10)
def sum4(impl=torch):
    a = impl.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    return a.sum(dim = 1, keepdim = True)

@testcase(name="max1: max along dim=0", score=10)
def max1(impl=torch):
    a = impl.Tensor([[1, 5, 2], [4, 3, 6]])
    vals, idxs = a.max(dim = 0)
    return (vals, idxs)

@testcase(name="max2: max along dim=1", score=10)
def max2(impl=torch):
    a = impl.Tensor([[1, 5, 2], [4, 3, 6]])
    vals, idxs = a.max(dim = 1)
    return (vals, idxs)

@testcase(name="max3: max with keepdim", score=10)
def max3(impl=torch):
    a = impl.Tensor([[1, 5, 2], [4, 3, 6]])
    vals, idxs = a.max(dim = 1, keepdim = True)
    return (vals, idxs)

@testcase(name="max4: max for higher dimension", score=10)
def max4(impl=torch):
    a = impl.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    return a.max(dim = 1)

@testcase(name="softmax1: softmax for tensor", score=10)
def softmax1(impl=torch):
    a = impl.Tensor([[1., 2., 3.], [2., 4., 6.]])
    return a.softmax(dim = 0)

@testcase(name="softmax2: softmax for bigger tensor", score=10)
def softmax2(impl=torch):
    a = impl.Tensor([[[1., -2., 3., 4., 5.], [6., 7., 8., 9., -10.]],
                     [[11., 12., 13., 14., 15.], [16., 17., 18., 19., -20.]],
                     [[21., 22., 23., 24., 25.], [26., -27., 28., 29., -30.]],
                     [[31., 32., -33., 34., 35.], [36., 37., 38., 39., -40.]]])
    return a.softmax(dim = 1)

def testsets_sum_max_softmax():
    sum1()
    sum2()
    sum3()
    sum4()
    max1()
    max2()
    max3()
    max4()
    softmax1()
    softmax2()

if __name__ == "__main__":
    print("Beginning grading part6")
    # set_debug_mode(True)
    testsets_sum_max_softmax()
    grader_summary("part6")