#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <sstream>
#include "tensor.h"
#include "meta.h"

namespace py = pybind11;
using shape_t = at::shape_t;
using Tensor = at::Tensor;
using stride_t = at::stride_t;
using dtype = at::dtype;
using Storage = at::Storage;
using slice_t = at::slice_t;

void parse_nested_list(const py::handle& obj, std::vector<at::dtype>& data, shape_t& shape, int depth);
slice_t py_slice_to_slice_t(const py::slice& s, ssize_t dim_size);
template <typename Tensor>
std::vector<slice_t> parse_index(const Tensor& self, const py::object &idx);
Tensor tensor_reshape_wrapper(const Tensor &self, py::args args, bool copy);
Tensor tensor_view_wrapper(const Tensor &self, py::args args);
py::object tensor_to_list(const at::Tensor& tensor);

PYBIND11_MODULE(clownpiece, m) {
    py::class_<at::Tensor, std::shared_ptr<at::Tensor>>(m, "TensorBaseImpl")
        .def(py::init<>())
        .def("__repr__",
            [](const at::Tensor &t) {
                std::ostringstream oss;
                // oss << "I AM CLOWNPIECE TENSOR!\n";
                oss << t;
                return oss.str();
            }
        )
        /*** Part Ø: Bindings for graderlib ***/
        .def_property_readonly("shape", &at::Tensor::get_shape)
        .def("data_at", &at::Tensor::get_data_at, py::arg("index"),
            "Get the data at the specified index. This is for graderlib use only.")
        .def("change_data_at", [](at::Tensor &self, int index, dtype value) {
            self.get_data_at(index) = value;
        }, py::arg("index"), py::arg("value"),
            "Change the data at the specified index. This is for graderlib use only.")
        .def("tolist", &tensor_to_list)
        .def(py::pickle(
            // __getstate__: 序列化
            [](const at::Tensor& t) {
                // 提取 shape
                shape_t shape = t.get_shape();
                // 提取数据
                const Storage& storage = t.get_storage();
                std::vector<at::dtype> data(storage.size);
                std::memcpy(data.data(), storage.data.get(), storage.size * sizeof(at::dtype));

                // 返回tuple: (shape, data list)
                return py::make_tuple(shape, data);
            },
            // __setstate__: 反序列化
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for TensorBase!");
                shape_t shape = t[0].cast<shape_t>();
                std::vector<at::dtype> data = t[1].cast<std::vector<at::dtype>>();

                return at::Tensor(shape, data);
            }
        ))

        /*** Part I: constructors, assignments, destructors, and item() ***/
        /* constructors */
        .def(py::init<at::dtype>(), py::arg("value"))
        // .def(py::init<const shape_t &>(), py::arg("shape"))
        .def(py::init<const shape_t &, at::dtype>(), py::arg("shape"), py::arg("value"))
        .def(py::init<const shape_t &, std::function<at::dtype()>>(), py::arg("shape"), py::arg("generator"))
        .def(py::init<const shape_t &, at::vec<at::dtype>>(), py::arg("shape"), py::arg("data"))
        .def(py::init<const shape_t &, at::stride_t, int, at::Storage>(), py::arg("shape"), py::arg("stride"), py::arg("offset"), py::arg("storage"))
        .def(py::init<const at::Tensor&>(), py::arg("other"))

        // Initialize from a Python list given shape
        .def(py::init([](const py::list& list) {
            at::vec<at::dtype> data;
            shape_t shape;
            parse_nested_list(list, data, shape, 0);
            return new at::Tensor(shape, data);
        }), py::arg("data"))

        /* assignments */
        .def("__copy__", &at::Tensor::clone, "Create a copy of the tensor")
        .def("__deepcopy__", &at::Tensor::clone, "Create a deep copy of the tensor")

        /* item() */
        .def("item", &at::Tensor::item, "Get the single value of a singleton tensor")



        /*** Part II: utils, clone, make contiguous and copy_ ***/
        .def("dim", &at::Tensor::dim)
        .def("size", [](const Tensor &self) {
            return self.size();
        }, "Get the size of the tensor")
        .def("size", [](const Tensor &self, int dim) {
            return self.size(dim);
        }, py::arg("dim"), "Get the size of a specific dimension")
        .def("is_contiguous", &at::Tensor::is_contiguous, "Check if the tensor is contiguous")

        .def("clone", &at::Tensor::clone)
        .def("contiguous", &at::Tensor::contiguous, "Make the tensor contiguous")
        .def("copy_", &at::Tensor::copy_, py::arg("other"), "Copy data from another tensor")
        .def("scatter_", &at::Tensor::scatter_,
             py::arg("dim"), py::arg("index"), py::arg("src"),
             "Scatter values from src tensor to this tensor along a specified dimension using index tensor")



        /*** Part III: subscriptor ***/
        .def("__getitem__", [](const at::Tensor &self, const at::veci &index) {
            return self[index];
        }, "Get a tensor using a vector of indices")
        .def("__getitem__", [](const at::Tensor &self, int index) {
            return self[index];
        }, "Get a tensor using a single index")
        .def("__getitem__", [](const at::Tensor &self, py::object idx) {
            auto slices = parse_index(self, idx);
            return self[slices];
        }, "Get a sliced tensor (supports int, slice, tuple of int/slice)")

        .def("__setitem__", [](at::Tensor& self, const at::veci& index, float value) {
            self[index] = value;
        }, "Set a tensor value using a vector of indices")
        .def("__setitem__", [](at::Tensor& self, int index, float value) {
            self[index] = value;
        }, "Set a tensor value using a single index")
        .def("__setitem__", [](at::Tensor& self, py::object idx, float value) {
            auto slices = parse_index(self, idx);
            self[slices] = value;
        }, "Set a sliced tensor (supports int, slice, tuple of int/slice)")
        .def("__setitem__", [](at::Tensor& self, int index, const at::Tensor& other) {
            for(int i = 0; i < other.numel(); i++) {
                self[index].data_at(i) = other.data_at(i);
            }
        }, "Set a sliced tensor using a another sliced tensor")
        .def("__setitem__", [](at::Tensor& self, py::object idx, const at::Tensor& other){
            auto slices = parse_index(self, idx);
            self[slices] = other;
        }, "Set a sliced tensor with another tensor (supports int, slice, tuple of int/slice)")



        /*** Part IV: Element-wise Binary and Unary Operators ***/
        .def(py::self <= py::self)
        .def(py::self <  py::self)
        .def(py::self >= py::self)
        .def(py::self >  py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__neg__", [](const at::Tensor &a) {
            return -a;
        })
        .def("__add__", [](const at::Tensor &a, const at::Tensor &b) {
            return a + b;
        })
        .def("__add__", [](const at::Tensor &a, at::dtype &b) {
            return a + b;
        })
        .def("__radd__", [](const at::Tensor &a, at::dtype &b) {
            return a + b;
        })
        .def("__sub__", [](const at::Tensor &a, const at::Tensor &b) {
            return a - b;
        })
        .def("__sub__", [](const at::Tensor &a, at::dtype &b) {
            return a - b;
        })
        .def("__rsub__", [](const at::Tensor &a, at::dtype &b) {
            return b - a;
        })
        .def("__mul__", [](const at::Tensor &a, const at::Tensor &b) {
            return a * b;
        })
        .def("__mul__", [](const at::Tensor &a, at::dtype &b) {
            return a * b;
        })
        .def("__rmul__", [](const at::Tensor &a, at::dtype &b) {
            return a * b;
        })
        .def("__truediv__", [](const at::Tensor &a, const at::Tensor &b) {
            return a / b;
        })
        .def("__truediv__", [](const at::Tensor &a, at::dtype &b) {
            return a / b;
        })
        .def("__rtruediv__", [](const at::Tensor &a, at::dtype &b) {
            return b / a;
        })
        .def("sign", &at::Tensor::sign)
        .def("abs", &at::Tensor::abs)
        .def("__abs__", &at::Tensor::abs)
        .def("sin", &at::Tensor::sin)
        .def("cos", &at::Tensor::cos)
        .def("tanh", &at::Tensor::tanh)
        .def("clamp", &at::Tensor::clamp, py::arg("min"), py::arg("max"))
        .def("log", &at::Tensor::log)
        .def("exp", &at::Tensor::exp)
        .def("pow", &at::Tensor::pow, py::arg("exponent"))
        .def("sqrt", &at::Tensor::sqrt)



        /*** Part V: Matrix Multiplication ***/
        .def("matmul", &at::matmul, py::arg("other"), "Matrix multiplication of two tensors")
        .def("__matmul__", [](const at::Tensor &a, const at::Tensor &b) {
            return a ^ b;
        })


        /*** Part VI: Reduction and Normalization Operations ***/
        .def("sum", &at::Tensor::sum, py::arg("dim"), py::arg("keepdim") = false, "Sum over a dimension")
        .def("max", &at::Tensor::max, py::arg("dim"), py::arg("keepdim") = false, "Get the maximum value and its index over a dimension")
        .def("softmax", &at::Tensor::softmax, py::arg("dim"), "Compute the softmax over a dimension")



        /*** Part VII: Shape Manipulation ***/
        .def("reshape", &tensor_reshape_wrapper, py::kw_only(), py::arg("copy") = false,
             "Reshape the tensor to a new shape, optionally copying data")
        .def("view", &tensor_view_wrapper,
             "Return a view of the tensor with a new shape")
        .def("transpose", &at::Tensor::transpose, py::arg("dim0"), py::arg("dim1"),
         "Transpose the tensor along two dimensions")
        // 增加.T属性
        .def_property_readonly("T", [](const at::Tensor &self) {
            // 通常 2D tensor 的 .T 是 swap 0,1 轴；更高维可按需扩展
            return self.transpose(0, 1);
        }, "Matrix transpose (swap axes 0 and 1)")
        .def_static("ones", [](std::vector<int> shape){
            return at::ones(shape);
        })
        .def_static("ones_like", [](const at::Tensor &self) {
            return at::ones_like(self);
        }, "Create a tensor of ones with the same shape and type as another tensor")
        .def_static("zeros", [](std::vector<int> shape){
            return at::zeros(shape);
        })
        .def_static("zeros_like", [](const at::Tensor &self) {
            return at::zeros_like(self);
        }, "Create a tensor of zeros with the same shape and type as another tensor")

        ;

    /*** Part II: utils, clone, make contiguous and copy_ ***/
    m.def("numel", [](at::Tensor& self) {
        return self.numel();
    }, py::arg("self"), "Calculate the number of elements in a tensor given its shape");

    /*** Part IV: Element-wise Binary and Unary Operators ***/
    m.def("__eq__", [](const at::Tensor &a, const at::Tensor &b) {
        return a == b;
    }, py::arg("a"), py::arg("b"), "Element-wise equality comparison");
    m.def("__ne__", [](const at::Tensor &a, const at::Tensor &b) {
        return a != b;
    }, py::arg("a"), py::arg("b"), "Element-wise inequality comparison");
    m.def("__lt__", [](const at::Tensor &a, const at::Tensor &b) {
        return a < b;
    }, py::arg("a"), py::arg("b"), "Element-wise less than comparison");
    m.def("__le__", [](const at::Tensor &a, const at::Tensor &b) {
        return a <= b;
    }, py::arg("a"), py::arg("b"), "Element-wise less than or equal to comparison");
    m.def("__gt__", [](const at::Tensor &a, const at::Tensor &b) {
        return a > b;
    }, py::arg("a"), py::arg("b"), "Element-wise greater than comparison");
    m.def("__ge__", [](const at::Tensor &a, const at::Tensor &b) {
        return a >= b;
    }, py::arg("a"), py::arg("b"), "Element-wise greater than or equal to comparison");


    m.def("sign", [](const at::Tensor& t) {
        return t.sign();
    });
    m.def("sin", [](const at::Tensor& t) {
        return t.sin();
    });
    m.def("cos", [](const at::Tensor& t) {
        return t.cos();
    });
    m.def("tanh", [](const at::Tensor& t) {
        return t.tanh();
    });
    m.def("clamp", [](const at::Tensor& t, const at::dtype& min, const at::dtype& max) {
        return t.clamp(min, max);
    }, py::arg("t"), py::arg("min"), py::arg("max"));
    m.def("log", [](const at::Tensor& t) {
        return t.log();
    }, py::arg("t"));
    m.def("exp", [](const at::Tensor& t) {
        return t.exp();
    });
    m.def("pow", [](const at::Tensor& t, at::dtype exponent) {
        return t.pow(exponent);
    }, py::arg("t"), py::arg("exponent"));
    m.def("sqrt", [](const at::Tensor& t) {
        return t.sqrt();
    });

    /*** Part VI: Reduction and Normalization Operations ***/
    m.def("dot", [](const at::Tensor &a, const at::Tensor &b) {
        return a ^ b;
    }, py::arg("a"), py::arg("b"), "Dot product of two tensors");
    m.def("sum", &at::Tensor::sum, py::arg("t"), py::arg("dim"), py::arg("keepdim") = false, "Sum over a dimension");
    m.def("max", &at::Tensor::max, py::arg("t"), py::arg("dim"), py::arg("keepdim") = false, "Get the maximum value and its index over a dimension");
    m.def("softmax", &at::Tensor::softmax, py::arg("t"), py::arg("dim"), "Compute the softmax over a dimension");

    /*** Part VII: Shape Manipulation ***/
    m.def("permute", &at::Tensor::permute, py::arg("t"), py::arg("dims"), "Permute the dimensions of the tensor");
    m.def("transpose", &at::Tensor::transpose, py::arg("t"), py::arg("dim0"), py::arg("dim1"), "Transpose the tensor along two dimensions");
    m.def("narrow", &at::Tensor::narrow, py::arg("t"), py::arg("dim"), py::arg("start"), py::arg("length"), py::arg("copy") = false, "Narrow the tensor along a dimension");
    m.def("chunk", &at::Tensor::chunk, py::arg("t"), py::arg("chunks"), py::arg("dim"), "Split the tensor into chunks along a dimension");
    m.def("split", [](const at::Tensor &self, int split_size, int dim = 0) {
        return self.split(dim, split_size);
    }, py::arg("t"), py::arg("split_size"), py::arg("dim") = 0, "Split the tensor into chunks of a given size along a dimension");
    m.def("split", [](const at::Tensor &self, const shape_t &split_sizes, int dim = 0) {
        return self.split(dim, split_sizes);
    }, py::arg("t"), py::arg("split_sizes"), py::arg("dim") = 0, "Split the tensor into chunks of specified sizes along a dimension");
    m.def("stack", [](const std::vector<at::Tensor> &tensors, int dim = 0) {
        return at::Tensor().stack(tensors, dim);
    }, py::arg("tensors"), py::arg("dim") = 0, "Stack a list of tensors along a new dimension");
    m.def("cat", [](const std::vector<at::Tensor> &tensors, int dim = 0) {
        return at::Tensor().cat(tensors, dim);
    }, py::arg("tensors"), py::arg("dim") = 0, "Concatenate a list of tensors along a dimension");
    m.def("squeeze", [](const at::Tensor &self, int dim = -1) {
        return self.squeeze(dim);
    }, py::arg("t"), py::arg("dim"), "Remove a dimension of size 1 from the tensor");
    m.def("unsqueeze", [](const at::Tensor &self, int dim = 0) {
        return self.unsqueeze(dim);
    }, py::arg("t"), py::arg("dim"), "Add a dimension of size 1 to the tensor");
    m.def("broadcast_to", [](const at::Tensor &self, const shape_t &shape) {
        return self.broadcast_to(shape);
    }, py::arg("t"), py::arg("shape"), "Broadcast the tensor to a new shape");
    m.def("broadcast_tensors", [](const at::Tensor &lhs, const at::Tensor &rhs) {
        return at::Tensor().broadcast(lhs, rhs);
    }, py::arg("lhs"), py::arg("rhs"), "Broadcast two tensors to a common shape");
    m.def("broadcast_tensors", [](const std::vector<at::Tensor> &tensors) {
        return at::Tensor().broadcast(tensors);
    }, py::arg("tensors"), "Broadcast a list of tensors to a common shape");

    /*** Part VIII: Other Helper Constuctors: ***/
    m.def("to_singleton_tensor", [](const at::dtype &value, int dim) {
        return at::to_singleton_tensor(value, dim);
    }, py::arg("value"), py::arg("dim"), "Create a singleton tensor with a single value and specified dimension");
    m.def("ones", [](const shape_t &shape) {
        return at::ones(shape);
    }, py::arg("shape"));
    m.def("ones_like", [](const at::Tensor &self) {
        return at::ones_like(self);
    }, py::arg("self"), "Create a tensor of ones with the same shape and type as another tensor");
    m.def("zeros", [](const shape_t &shape) {
        return at::zeros(shape);
    }, py::arg("shape"));
    m.def("zeros_like", [](const at::Tensor &self) {
        return at::zeros_like(self);
    }, py::arg("self"), "Create a tensor of zeros with the same shape and type as another tensor");
    m.def("randn", [](const shape_t &shape) {
        return at::randn(shape);
    }, py::arg("shape"));
    m.def("randn_like", [](const at::Tensor &self) {
        return at::randn_like(self);
    }, py::arg("self"), "Create a tensor of random values with the same shape and type as another tensor");
    m.def("empty", [](const shape_t &shape) {
        return at::empty(shape);
    }, py::arg("shape"), "Create an empty tensor with the specified shape");
    m.def("empty_like", [](const at::Tensor &self) {
        return at::empty_like(self);
    }, py::arg("self"), "Create an empty tensor with the same shape and type as another tensor");
    m.def("arange", [](at::dtype start, at::dtype end, int step = 1) {
        return at::arange(start, end, step);
    }, py::arg("start"), py::arg("end"), py::arg("step") = 1, "Create a tensor with a range of values from start to end with a specified step");
    m.def("range", [](at::dtype start, at::dtype end, int step = 1) {
        return at::range(start, end, step);
    }, py::arg("start"), py::arg("end"), py::arg("step") = 1, "Create a tensor with a range of values from start to end with a specified step");
    m.def("linspace", [](at::dtype start, at::dtype end, int num_steps) {
        return at::linspace(start, end, num_steps);
    }, py::arg("start"), py::arg("end"), py::arg("num_steps"), "Create a tensor with linearly spaced values from start to end with a specified number of steps");
}

/* Utils */

// 递归解析嵌套列表的工具函数
void parse_nested_list(const py::handle& obj, std::vector<at::dtype>& data, shape_t& shape, int depth = 0) {
    if (py::isinstance<py::list>(obj)) {
        py::list list = py::cast<py::list>(obj);
        if (list.empty()) {
            throw std::runtime_error("Empty lists are not supported for tensor initialization.");
        }

        // Only add a new dimension if we're going deeper than before
        if (depth >= shape.size()) {
            shape.push_back(list.size());
        } else if (list.size() != static_cast<size_t>(shape[depth])) {
            throw std::runtime_error("Inconsistent list lengths in nested lists.");
        }

        // Recursively process each item
        for (const auto& item : list) {
            parse_nested_list(item, data, shape, depth + 1);
        }
    } else {
        // Leaf node: add the scalar value
        data.push_back(py::cast<at::dtype>(obj));
    }
}

slice_t py_slice_to_slice_t(const py::slice& s, ssize_t dim_size) {
    py::ssize_t start, stop, step, slicelength;
    if (!s.compute(dim_size, &start, &stop, &step, &slicelength)) {
        throw std::runtime_error("Invalid Python slice");
    }
    if (step != 1) {
        throw std::runtime_error("Only step=1 is supported");
    }
    return slice_t(static_cast<int>(start), static_cast<int>(stop));
}

// 递归处理索引对象
template <typename Tensor>
std::vector<slice_t> parse_index(const Tensor& self, const py::object &idx) {
    std::vector<slice_t> slices;

    if (py::isinstance<py::tuple>(idx)) {
        auto t = idx.cast<py::tuple>();
        size_t nd = self.dim();
        for (size_t i = 0; i < t.size(); ++i) {
            auto item = t[i];
            if (py::isinstance<py::int_>(item)) {
                int index = item.cast<int>();
                slices.emplace_back(index, index + 1);
            } else if (py::isinstance<py::slice>(item)) {
                slices.push_back(py_slice_to_slice_t(item.cast<py::slice>(), self.size(i)));
            } else if (py::isinstance<py::tuple>(item)) {
                auto subt = item.cast<py::tuple>();
                if (subt.size() == 2) {
                    slices.emplace_back(subt[0].cast<int>(), subt[1].cast<int>());
                } else {
                    throw std::runtime_error("Invalid tuple size for slice");
                }
            } else {
                throw std::runtime_error("Invalid index type in tuple");
            }
        }
    } else if (py::isinstance<py::slice>(idx)) {
        slices.push_back(py_slice_to_slice_t(idx.cast<py::slice>(), self.size(0)));
    } else if (py::isinstance<py::int_>(idx)) {
        int index = idx.cast<int>();
        slices.emplace_back(index, index + 1);
    } else if (py::isinstance<py::tuple>(idx)) {
        auto subt = idx.cast<py::tuple>();
        if (subt.size() == 2) {
            slices.emplace_back(subt[0].cast<int>(), subt[1].cast<int>());
        } else {
            throw std::runtime_error("Invalid tuple size for slice");
        }
    } else {
        throw std::runtime_error("Unsupported index type");
    }
    return slices;
}

Tensor tensor_reshape_wrapper(const Tensor &self, py::args args, bool copy = false) {
    std::vector<int> shape;
    if (args.size() == 1 && py::isinstance<py::sequence>(args[0])) {
        shape = args[0].cast<std::vector<int>>();
    } else {
        for (auto item : args) {
            shape.push_back(item.cast<int>());
        }
    }
    return self.reshape(shape, copy);
}

Tensor tensor_view_wrapper(const Tensor &self, py::args args) {
    std::vector<int> shape;
    if (args.size() == 1 && py::isinstance<py::sequence>(args[0])) {
        shape = args[0].cast<std::vector<int>>();
    } else {
        for (auto item : args) {
            shape.push_back(item.cast<int>());
        }
    }
    return self.view(shape);
}

// 递归辅助函数
py::object tensor_to_list_recursive(const at::Tensor& tensor, int dim, std::vector<int> indices) {
    // 如果是最后一维
    if (dim == tensor.dim() - 1) {
        py::list l;
        int len = tensor.size(dim);
        for (int i = 0; i < len; ++i) {
            indices.push_back(i);
            // 计算flat index
            int flat_idx = tensor.get_offset();
            const auto& stride = tensor.get_stride();
            for (int d = 0; d < tensor.dim(); ++d)
                flat_idx += indices[d] * stride[d];
            l.append(py::float_(tensor.get_storage().data[flat_idx]));
            indices.pop_back();
        }
        return l;
    } else {
        py::list l;
        int len = tensor.size(dim);
        for (int i = 0; i < len; ++i) {
            indices.push_back(i);
            l.append(tensor_to_list_recursive(tensor, dim + 1, indices));
            indices.pop_back();
        }
        return l;
    }
}

// 对外接口
py::object tensor_to_list(const at::Tensor& tensor) {
    if (tensor.dim() == 0) {
        // 标量
        return py::float_(tensor.item());
    }
    return tensor_to_list_recursive(tensor, 0, {});
}