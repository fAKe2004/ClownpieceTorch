#include "tensor.h"
#include <iostream>
#include <vector>
#include <cmath>

// Helper function to print tensors for easier debugging
void print_tensor(const std::string& name, const at::Tensor& t) {
    std::cout << name << ":\n" << t << std::endl << std::endl;
}

void test_tensor_creation() {
    std::cout << "--- Test: Tensor Creation ---" << std::endl;
    at::Tensor a = at::ones({2, 3});
    at::Tensor b = at::zeros({2, 3});
    at::Tensor c = at::randn({2, 3});
    at::Tensor d = at::arange(0, 6, 1).reshape({2, 3});
    at::Tensor e = at::linspace(0, 1, 5);

    print_tensor("Tensor a (ones)", a);
    print_tensor("Tensor b (zeros)", b);
    print_tensor("Tensor c (randn)", c);
    print_tensor("Tensor d (arange)", d);
    print_tensor("Tensor e (linspace)", e);
}

void test_tensor_operations() {
    std::cout << "--- Test: Tensor Operations ---" << std::endl;
    at::Tensor a = at::Tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    at::Tensor b = at::Tensor({2, 3}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    at::dtype scalar = 2.0;

    print_tensor("Tensor a", a);
    print_tensor("Tensor b", b);

    print_tensor("a + b", a + b);
    print_tensor("a - b", a - b);
    print_tensor("a * b", a * b);
    print_tensor("a / b", a / b);
    print_tensor("a + scalar", a + scalar);
    print_tensor("scalar + a", scalar + a);
    print_tensor("a - scalar", a - scalar);
    print_tensor("scalar - a", scalar - a);
    print_tensor("a * scalar", a * scalar);
    print_tensor("scalar * a", scalar * a);
    print_tensor("a / scalar", a / scalar);
    print_tensor("scalar / a", scalar / a);
    print_tensor("-a", -a);
}

void test_tensor_reductions() {
    std::cout << "--- Test: Tensor Reductions ---" << std::endl;
    at::Tensor a = at::Tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    print_tensor("Tensor a", a);
    print_tensor("sum(a, dim=0)", a.sum(0));
    print_tensor("sum(a, dim=1)", a.sum(1));
    print_tensor("max(a, dim=0)", a.max(0).first);
    print_tensor("max(a, dim=0) indices", a.max(0).second);
    print_tensor("max(a, dim=1)", a.max(1).first);
    print_tensor("max(a, dim=1) indices", a.max(1).second);
}

void test_tensor_broadcasting() {
    std::cout << "--- Test: Tensor Broadcasting ---" << std::endl;
    at::Tensor a = at::Tensor({2, 1}, {1.0, 2.0});
    at::Tensor b = at::Tensor({1, 3}, {3.0, 4.0, 5.0});
    at::Tensor c = a + b;

    print_tensor("Tensor a", a);
    print_tensor("Tensor b", b);
    print_tensor("a + b (broadcasted)", c);
}

void test_tensor_indexing() {
    std::cout << "--- Test: Tensor Indexing ---" << std::endl;
    at::Tensor a = at::Tensor({3, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    print_tensor("Tensor a", a);
    print_tensor("a[0]", a[0]);
    print_tensor("a[1, 1]", a[at::veci({1, 1})]);
    print_tensor("a[:, 1]", a[{at::slice_t(0, 3), at::slice_t(1, 2)}]);
}

void test_tensor_reshape_and_transpose() {
    std::cout << "--- Test: Tensor Reshape and Transpose ---" << std::endl;
    at::Tensor a = at::Tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    print_tensor("Tensor a", a);
    print_tensor("a.reshape({3, 2})", a.reshape({3, 2}));
    print_tensor("a.transpose(0, 1)", a.transpose(0, 1));
}

void test_tensor_matmul() {
    std::cout << "--- Test: Tensor Matmul ---" << std::endl;
    at::Tensor a = at::Tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    at::Tensor b = at::Tensor({3, 2}, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    at::Tensor c = at::matmul(a, b);

    print_tensor("Tensor a", a);
    print_tensor("Tensor b", b);
    print_tensor("a.matmul(b)", c);
}

void test_tensor_math_functions() {
    std::cout << "--- Test: Tensor Math Functions ---" << std::endl;
    at::Tensor a = at::Tensor({2, 3}, {-1.0, -2.0, -3.0, 4.0, 5.0, 6.0});

    print_tensor("Tensor a", a);
    print_tensor("a.abs()", a.abs());
    print_tensor("a.sin()", a.sin());
    print_tensor("a.exp()", a.exp());
    print_tensor("a.log()", a.log(2.71828));
    print_tensor("a.sqrt()", a.sqrt());
    print_tensor("a.pow(2)", a.pow(2));
    print_tensor("a.clamp(0, 5)", a.clamp(0, 5));
}

int main() {
    test_tensor_creation();
    test_tensor_operations();
    test_tensor_reductions();
    test_tensor_broadcasting();
    test_tensor_indexing();
    test_tensor_reshape_and_transpose();
    test_tensor_matmul();
    test_tensor_math_functions();

    return 0;
}
