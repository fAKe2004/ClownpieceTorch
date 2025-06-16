#include "tensor.h"
#include <cmath>
#include <string>


namespace at {

  /*
    utils
  */
  int ceil_div(int num, int den) {
    return (num + den - 1) / den;
  }

  int calc_numel(const shape_t& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  }

  stride_t calc_stride(const shape_t& shape) {
    stride_t stride(shape.size());
    int prod = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        stride[i] = prod;
        prod *= shape[i];
    }
    return stride;
  }

  slice_t normalize_slice(slice_t slice, int len) {
    if (slice.first < 0)
      slice.first += len;
    if (slice.second < 0)
      slice.second += len;
    
    slice.first = std::max(slice.first, 0);
    slice.second = std::min(slice.second, len);

    if (slice.first >= slice.second) //empty slice
      // throw std::out_of_range("normalize: slice out of range");
      slice = slice_t(0, 0);

    return slice;
  }

  int normalize_index(int index, int len) {
    if (index < 0)
      index += len;
    if (index < 0 || index >= len)
      throw std::out_of_range("normalize: index out of range");
    return index;
  }

  bool check_shape_match(const shape_t& a, const shape_t& b) {
    if (a.size() != b.size())
      return false;
    for (int i = 0; i < (int)a.size(); i++)
      if (a[i] != b[i])
        return false;
    return true;
  }

  shape_t broadcast_shape(const shape_t& a, const shape_t& b) {
    int max_dim = std::max(a.size(), b.size());
    shape_t result(max_dim);
    for (int i = 0; i < max_dim; i++) {
      int a_size = (i < (int)a.size()) ? a[a.size() - 1 - i] : 1;
      int b_size = (i < (int)b.size()) ? b[b.size() - 1 - i] : 1;
      if (a_size != 1 && b_size != 1 && a_size != b_size)
        throw std::invalid_argument("broadcast_shape: shape mismatch");
      result[max_dim - 1 - i] = std::max(a_size, b_size);
    }
    return result;
  }

  template<class T>
  vec<T> concat_vec(const vec<T>& a, const vec<T>& b) {
    vec<T> result(a.size() + b.size());
    std::copy(a.begin(), a.end(), result.begin());
    std::copy(b.begin(), b.end(), result.begin() + a.size());
    return result;
  }

  // print vector int
  std::ostream& operator<<(std::ostream& os, const shape_t& shape) {
    os << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      os << shape[i];
      if (i < shape.size() - 1)
        os << ", ";
    }
    os << ")";
    return os;
  }

  int print_tensor_data_recursive(std::ostream& os, const Tensor& tensor, int dim_index, int data_index, std::string prefix) {
    if (tensor.dim() == 0) {
      if (tensor.numel() == 0)
        os << "[]";
      else
        os << tensor.data_at(0);
      return 0;
    }
    os << "[";
    if (dim_index == tensor.dim() - 1 || tensor.dim() == 0) {
      for (int i = 0; i < tensor.size(dim_index); ++i) {
        os << tensor.data_at(data_index++);
        if (i < tensor.size(dim_index) - 1)
            os << ", ";
      }
    } else {

      for (int i = 0; i < tensor.size(dim_index); ++i) {
        if (i > 0)
          os << "\n" << prefix;
        data_index = print_tensor_data_recursive(os, tensor, dim_index + 1, data_index, prefix + " ");
        if (i < tensor.size(dim_index) - 1)
          os << ",";
      }
    }
    os << "]";
    return data_index;
  }






  /*
    Beginning of Tensor implementation
  */

  void Tensor::post_init() {
    numel_ = calc_numel(shape_);
    if (numel_ == 0) // if the tensor ever becomes empty, set the storage to empty
      storage_ = Storage();
    if (storage_.data == nullptr)
      numel_ = 0;
    dim_ = shape_.size();
    shape_prod_ = calc_stride(shape_);
    is_contiguous_ = true;
    for (int i = 0; i < dim_; ++i) {
      if (stride_[i] != shape_prod_[i]) {
        is_contiguous_ = false;
        break;
      }
    }
  }

  int Tensor::real_numel() const {
    // count numel in physical storage, removing duplication from broadcast
    if (!numel_)
      return 0;

    int prod = 1;
    for (int i = 0; i < dim_; ++i) {
      if (shape_[i] > 0 && stride_[i] != 0)
        prod *= shape_[i];
    }
    return prod;
  }

  Tensor apply_unary_op(const Tensor& input, const std::function<dtype(dtype)>& op) {
    Tensor output = input.empty_like();

    for (int i = 0; i < input.numel(); i++)
      output.data_at(i) = op(input.data_at(i));

    return output;
  }

  Tensor apply_binary_op(const Tensor& lhs, const Tensor& rhs, const std::function<dtype(dtype, dtype)>& op) {
    // if (!check_shape_match(lhs.shape_, rhs.shape_))
    //   throw std::invalid_argument("apply_binary_op: shape mismatch");
    Tensor lhs_b, rhs_b;
    std::tie(lhs_b, rhs_b) = Tensor::broadcast(lhs, rhs);

    Tensor output = lhs_b.empty_like();
    for (int i = 0; i < lhs_b.numel(); i++) {
      output.data_at(i) = op(lhs_b.data_at(i), rhs_b.data_at(i));
    }

    return output;
  }
  
  Tensor apply_binary_op(const Tensor& tensor, dtype scalar, const std::function<dtype(dtype, dtype)>& op) {
    Tensor output = tensor.empty_like();

    for (int i = 0; i < tensor.numel(); i++) {
      output.data_at(i) = op(tensor.data_at(i), scalar);
    }

    return output;
  }

  Tensor apply_binary_op(dtype scalar, const Tensor& tensor, const std::function<dtype(dtype, dtype)>& op) {
    Tensor output = tensor.empty_like();

    for (int i = 0; i < tensor.numel(); i++) {
      output.data_at(i) = op(scalar, tensor.data_at(i));
    }

    return output;
  }

  Tensor apply_along_axis(const Tensor& input, int dim, const std::function<dtype(const vec<dtype>&)>& op) {
    dim = normalize_index(dim, input.dim_);
    shape_t output_shape = input.shape_; 
    output_shape[dim] = 1;

    int dim_size = input.size(dim);
    vec<dtype> buffer(dim_size);

    Tensor input_T = input.transpose(dim, -1).reshape({-1, dim_size});
    shape_t output_T_shape = output_shape; 
    std::swap(output_T_shape[dim], output_T_shape.back());
    Tensor output_T = Tensor(output_T_shape);

    for (int i = 0; i < input_T.size(0); i++) {
      for (int j = 0; j < dim_size; j++) {
        buffer[j] = input_T.data_at(i * dim_size + j);
      }
      output_T.data_at(i) = op(buffer);
    }

    Tensor output = output_T.transpose(dim, -1);
    return output; // output dim is not squeezed
  }

  std::pair<Tensor, Tensor> apply_along_axis(
    const Tensor& input, 
    int dim, 
    const std::function<std::pair<dtype, dtype>(const vec<dtype>&)>& op) {
    dim = normalize_index(dim, input.dim_);
    shape_t output_shape = input.shape_; 
    output_shape[dim] = 1;

    int dim_size = input.size(dim);
    vec<dtype> buffer(dim_size);

    Tensor input_T = input.transpose(dim, -1).reshape({-1, dim_size});
    shape_t output_T_shape = output_shape; 
    std::swap(output_T_shape[dim], output_T_shape.back());
    Tensor output_T_1 = Tensor(output_T_shape);
    Tensor output_T_2 = Tensor(output_T_shape);

    for (int i = 0; i < input_T.size(0); i++) {
      for (int j = 0; j < dim_size; j++) {
        buffer[j] = input_T.data_at(i * dim_size + j);
      }
      std::tie(output_T_1.data_at(i), output_T_2.data_at(i)) = op(buffer);
    }

    Tensor output_1 = output_T_1.transpose(dim, -1);
    Tensor output_2 = output_T_2.transpose(dim, -1);
    return std::make_pair(output_1, output_2); // output dim is not squeezed
  }

  // physical index of the index-th element in flattened logic order
  int Tensor::index_at(int index) const {
    if (index < 0 || index >= numel_)
      throw std::out_of_range("index out of range");

    if (is_contiguous())
      return offset_ + index;

    int element_offset = offset_;
    for (int d = 0; d < dim_; d++) {
      int index_d = index / shape_prod_[d];
      index = index % shape_prod_[d];
      element_offset += index_d * stride_[d];
    }
    return element_offset;
  }

  // underlying data at the index-th element in flattened logic order
  dtype& Tensor::data_at(int index) const {
    return storage_[index_at(index)];
  }

  // induce shape from purposed shape with placeholder -1.
  shape_t Tensor::induce_shape(const shape_t& shape) const {

    shape_t new_shape = shape;
    int neg1_at = -1;
    int prod = 1;
    for (int i = 0; i < (int)new_shape.size(); ++i) {
      if (new_shape[i] == -1) {
        if (neg1_at != -1)
          throw std::invalid_argument("induce_shape: only one -1 is allowed in the shape");
        neg1_at = i;
      } else
        prod *= new_shape[i];
    }
    if (neg1_at == -1)
      return new_shape;

    if (prod == 0 || numel() % prod != 0)
      throw std::invalid_argument("induce_shape: either zero or not divisible");

    new_shape[neg1_at] = numel() / prod;
    return new_shape;
  }


  /*
    constructors and assignments
  */
  Tensor::Tensor() 
    : shape_(shape_t()), stride_(stride_t()), offset_(0), storage_(Storage()){
    post_init();
  }
  Tensor::Tensor(dtype value)
    : Tensor(shape_t(), value) {}
  Tensor::Tensor(const shape_t& shape) 
    : shape_(shape), stride_(calc_stride(shape)), offset_(0), storage_(Storage(calc_numel(shape))) {
    post_init();
  }
  Tensor::Tensor(const shape_t& shape, dtype value) 
    : shape_(shape), stride_(calc_stride(shape)), offset_(0), storage_(Storage(calc_numel(shape), value)) {
    post_init();
  }
  Tensor::Tensor(const shape_t& shape, std::function<dtype()> generator) 
    : shape_(shape), stride_(calc_stride(shape)), offset_(0), storage_(Storage(calc_numel(shape), generator)) {
    post_init();
  }
  Tensor::Tensor(const shape_t& shape, const vec<dtype>& data)
    : shape_(shape), stride_(calc_stride(shape)), offset_(0), storage_(Storage(data)) {
      post_init();
      if (numel() != (int)data.size())
        throw std::invalid_argument("Tensor: shape and data size mismatch");
    }
  Tensor::Tensor(const shape_t& shape, const stride_t& stride, int offset, Storage storage) 
    : shape_(shape), stride_(stride), offset_(offset), storage_(storage) {
    post_init();
    if (shape.size() != stride.size())
      throw std::invalid_argument("Tensor: shape and stride size mismatch");
    if (offset < 0 || (storage_.data != nullptr and offset + real_numel() > (int)storage_.size))
      throw std::out_of_range("Tensor: offset out of range");
  }

  Tensor::Tensor(const Tensor& other) = default;

  Tensor& Tensor::operator=(const Tensor& other) = default;

  Tensor& Tensor::operator=(dtype value) {
    if (numel_ != 1)
      throw std::invalid_argument("Tensor: only singleton tensor can be assigned");
    this->data_at(0) = value;
    return *this;
  }

  /* 
    destructor
  */
  Tensor::~Tensor() = default;


  /*
    convert to dtype value
    only valid for singleton tensor
  */
  dtype Tensor::item() const {
    if (numel_ != 1)
      throw std::invalid_argument("Tensor: item can only be called on singleton tensor");
    return data_at(0);
  }

  /*
    utils
  */

  int Tensor::numel() const {
    return numel_;
  }

  int Tensor::dim() const {
    return dim_;
  }

  veci Tensor::size() const {
    return shape_;
  }

  int Tensor::size(int dim) const {
    dim = normalize_index(dim, dim_);
    return shape_[dim];
  }

  bool Tensor::is_contiguous() const {
    return is_contiguous_;
  }


  /*
    clone, make contiguous, copy_ and scatter
  */
  Tensor Tensor::clone() const {
    if (is_contiguous())
      return Tensor(shape_, stride_, offset_, storage_.clone());
    else {
      Tensor cloned = Tensor(shape_);
      for (int i = 0; i < numel_; i++)
        cloned.data_at(i) = data_at(i);
      return cloned;
    }
  }

  Tensor Tensor::contiguous() const {
    if (is_contiguous())
      return *this;
    else
      return clone();
  }

  Tensor Tensor::copy_(const Tensor& other) const {
    if (this == &other)
      return *this;

    if (!check_shape_match(shape_, other.shape_))
      throw std::invalid_argument("Tensor: shape mismatch");

    for (int i = 0; i < numel_; i++)
      this->data_at(i) = other.data_at(i);

    return *this;
  }

  Tensor Tensor::scatter_(int dim, const Tensor& index, const Tensor& src) const {
    dim = normalize_index(dim, dim_);

    shape_t shape = shape_;
    shape.erase(shape.begin() + dim);
    if (!check_shape_match(shape, index.shape_) || !check_shape_match(shape, src.shape_))
      throw std::invalid_argument("Tensor: shape mismatch in scatter");

    Tensor transposed = this->transpose(dim, -1);
    int dim_size = transposed.size(-1);
    for (int i = 0; i < index.numel_; i++) {
      int index_val = round(index.data_at(i)) + 0.5;
      dtype src_val = src.data_at(i);
      transposed.data_at(i * dim_size + index_val) = src_val;
    }

    return *this;
  }


  /*
    subscriptor
  */

  Tensor Tensor::operator[](const vec<slice_t>& slices) const {
    if (slices.size() == 0 || (int)slices.size() > dim_)
    throw std::invalid_argument("Tensor: slices are empty or size exceeds dim");
    
    vec<slice_t> padded_slices = slices;
    for (int i = slices.size(); i < dim_; i++)
      padded_slices.push_back(std::make_pair(0, shape_[i]));

    int new_offset = offset_;
    shape_t new_shape = shape_t(dim_);

    for (int d = 0; d < dim_; d++) {
      slice_t slice = normalize_slice(padded_slices[d], shape_[d]);

      new_offset += slice.first * stride_[d];
      new_shape[d] = slice.second - slice.first;
    }
    return Tensor(new_shape, stride_, new_offset, storage_);
  }

  Tensor Tensor::operator[](slice_t slice) const {
    if (!dim_)
      throw std::invalid_argument("Tensor: slicing tensor with dim = 0");
    slice = normalize_slice(slice, shape_[0]);
    shape_t new_shape = shape_;
    new_shape[0] = slice.second - slice.first;
    int new_offset = offset_ + slice.first * stride_[0];

    return Tensor(new_shape, stride_, new_offset, storage_);
  }

  Tensor Tensor::operator[](const veci& index) const {
    if (index.size() == 0 || (int)index.size() > dim_)
      throw std::invalid_argument("Tensor: index is empty or size exceeds dim");
    vec<slice_t> slices = vec<slice_t>(index.size());
    for (int i = 0; i < (int)index.size(); i++) {
      slices[i].first = index[i];
      slices[i].second = index[i] + 1;
    }
    Tensor sliced = operator[](slices);
    // squeeze
    for (int i = 0; i < (int)index.size(); i++)
      sliced = sliced.squeeze(0);
    return sliced;
  }

  Tensor Tensor::operator[](int index) const {
    Tensor sliced = operator[](slice_t(index, index + 1));
    // squeeze
    sliced = sliced.squeeze(0);
    return sliced;
  }

  /*
    operators
  */
  Tensor Tensor::operator-() const {
    return apply_unary_op(*this, [](dtype x)->dtype { return -x; });
  }

  Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    
    return apply_binary_op(lhs, rhs, [](dtype x, dtype y)->dtype { return x + y; });
  }
  
  Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype x, dtype y)->dtype { return x - y; });
  }

  Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype x, dtype y)->dtype { return x * y; });
  }

  Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype x, dtype y)->dtype { return x / y; });
  }

  Tensor operator==(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype x, dtype y) -> dtype { return x == y; });
  }

  Tensor operator!=(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype x, dtype y) -> dtype { return x != y; });
  }
  Tensor operator<(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype x, dtype y) -> dtype { return x < y; });
  }

  Tensor operator<=(const Tensor& lhs, const Tensor& rhs) {
      return apply_binary_op(lhs, rhs, [](dtype x, dtype y) -> dtype { return x <= y; });
  }

  Tensor operator>=(const Tensor& lhs, const Tensor& rhs) {
      return apply_binary_op(lhs, rhs, [](dtype x, dtype y) -> dtype { return x >= y; });
  }

  Tensor operator>(const Tensor& lhs, const Tensor& rhs) {
      return apply_binary_op(lhs, rhs, [](dtype x, dtype y) -> dtype { return x > y; });
  }

  /*
    matrix multiplication
  */
  Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.dim() || !rhs.dim())
      return Tensor();
    
    // adjust vector to matrix
    Tensor l_mat = lhs, r_mat = rhs;
    bool lhs_padded = false, rhs_padded = false;
    if (l_mat.dim() == 1) {
      l_mat=l_mat.unsqueeze(0);
      lhs_padded = true;
    }

    if (r_mat.dim() == 1) {
      r_mat=r_mat.unsqueeze(0);
      r_mat=r_mat.transpose(-1, -2);
      rhs_padded = true;
    }

    r_mat = r_mat.transpose(-1, -2); // l_mat=(..., l, m) r=(..., n, m) for contiguity.

    int l = l_mat.size(-2), m = l_mat.size(-1), n = r_mat.size(-2);

    if (l_mat.size(-1) != r_mat.size(-1))
      throw std::invalid_argument("matmul: inner dimension mismatch");
    
    shape_t l_batch_shape = shape_t(l_mat.shape_.begin(), l_mat.shape_.end() - 2);
    shape_t r_batch_shape = shape_t(r_mat.shape_.begin(), r_mat.shape_.end() - 2);
    shape_t batch_shape = broadcast_shape(l_batch_shape, r_batch_shape);
    shape_t l_bm_shape = concat_vec(batch_shape, shape_t({l, m}));
    shape_t r_bm_shape = concat_vec(batch_shape, shape_t({n, m}));

    Tensor l_bm = l_mat.broadcast_to(l_bm_shape);
    Tensor r_bm = r_mat.broadcast_to(r_bm_shape);
    
    int batch_size = std::accumulate(batch_shape.begin(), batch_shape.end(), 1, std::multiplies<int>());
    
    // make contiguous if necessary
    l_bm = l_bm.reshape({-1, l, m});
    r_bm = r_bm.reshape({-1, n, m});

    Tensor output = Tensor({batch_size, l, n});

    for (int b = 0; b < batch_size; b++) {
      Tensor l_mat_unit = l_bm[b];
      Tensor r_mat_unit = r_bm[b];
      Tensor output_unit = output[b];
      for (int i = 0; i < l; i++) {
        // Tensor l_mat_row = l_mat_unit[i];
        for (int j = 0; j < n; j++) {
          // Tensor r_mat_col = r_mat_unit[j];
          dtype sum = 0;
          for (int k = 0; k < m; k++) {
            sum += l_mat_unit.data_at(i * m + k) * r_mat_unit.data_at(j * m + k);
            // sum += l_mat_row.data_at(k) * r_mat_col.data_at(k);
          }
          output_unit.data_at(i * n + j) = sum;
        }
      }
    }

    shape_t output_shape = concat_vec(batch_shape, {l, n});

    output = output.view(output_shape);
    if (lhs_padded && rhs_padded)
      output = output.view({}); // dot product
    else if (lhs_padded)
      output = output.squeeze(-2); // row vec * mat
    else if (rhs_padded)
      output = output.transpose(-1, -2).squeeze(-2); // mat * col vec
    
    return output;
  }

  Tensor operator^(const Tensor& lhs, const Tensor& rhs) {
    return matmul(lhs, rhs);
  }

  /*
    other mathematical operations
  */
  Tensor Tensor::sign() const {
    return apply_unary_op(*this, [](dtype x)-> dtype {
      if (x > 0)
        return 1.0;
      if (x < 0)
        return -1.0;
      return 0.0;
    });
  }

  Tensor Tensor::abs() const {
    return apply_unary_op(*this, [](dtype x)->dtype { return std::abs(x); });
  }
  Tensor abs(const Tensor& tensor) {
    return tensor.abs();
  }
  Tensor Tensor::sin() const {
    return apply_unary_op(*this, [](dtype x)->dtype { return std::sin(x); });
  }
  Tensor sin(const Tensor& tensor) {
    return tensor.sin();
  }
  Tensor Tensor::tanh() const {
    return apply_unary_op(*this, [](dtype x)->dtype { return std::tanh(x); });
  }
  Tensor tanh(const Tensor& tensor) {
    return tensor.tanh();
  }

  Tensor Tensor::clamp(dtype min, dtype max) const {
    return apply_unary_op(*this, [min, max](dtype x)->dtype { 
      return std::max(min, std::min(max, x)); 
    });
  }

  Tensor clamp(const Tensor& tensor, dtype min, dtype max) {
    return tensor.clamp(min, max);
  }

  Tensor Tensor::log() const {
    return apply_unary_op(*this, [](dtype x)->dtype { 
      return std::log(x); 
    });
  }

  Tensor log(const Tensor& tensor) {
    return tensor.log();
  }

  Tensor Tensor::exp() const {
    return apply_unary_op(*this, [](dtype x)->dtype { 
      return std::exp(x); 
    });
  }

  Tensor exp(const Tensor& tensor) {
    return tensor.exp();
  }

  Tensor Tensor::pow(dtype exponent) const {
    return apply_unary_op(*this, [exponent](dtype x)->dtype { 
      return std::pow(x, exponent); 
    });
  }

  Tensor pow(const Tensor& tensor, dtype exponent) {
    return tensor.pow(exponent);
  }

  Tensor Tensor::sqrt() const {
    return apply_unary_op(*this, [](dtype x)->dtype { 
      return std::sqrt(x); 
    });
  }

  Tensor sqrt(const Tensor& tensor) {
    return tensor.sqrt();
  }

  Tensor Tensor::sum(int dim, bool keepdim) const {
    std::function<dtype(const vec<dtype>&)> sum_op = [keepdim](const vec<dtype>& vec) -> dtype {
      dtype sum = 0;
      for (const auto& v : vec)
        sum += v;
      return sum;
    };

    Tensor output = apply_along_axis(*this, dim, sum_op);
    if (!keepdim)
      output.squeeze(dim);
    return output;
  }

  Tensor sum(const Tensor& tensor, int dim, bool keepdim) {
    return tensor.sum(dim, keepdim);
  }

  std::pair<Tensor, Tensor> Tensor::max(int dim, bool keepdim) const {
    std::function<std::pair<dtype, dtype>(const vec<dtype>&)> max_op = [keepdim](const vec<dtype>& vec) -> std::pair<dtype, dtype> {
      if (vec.empty())
        throw std::invalid_argument("Tensor: max on empty vector");
      dtype max_val = vec[0];
      dtype max_at = 0;
      for (int i = 1; i < (int)vec.size(); i++) {
        dtype v = vec[i];
        if (v > max_val) {
          max_val = v;
          max_at = i;
        }
      }
      return std::make_pair(max_val, max_at);
    };

    Tensor output_1, output_2;
    std::tie(output_1, output_2) = apply_along_axis(*this, dim, max_op);
    if (!keepdim) {
      output_1.squeeze(dim);
      output_2.squeeze(dim);
    }
    return std::make_pair(output_1, output_2);
  }

  std::pair<Tensor, Tensor> max(const Tensor& tensor, int dim, bool keepdim) {
    return tensor.max(dim, keepdim);
  }

  Tensor Tensor::softmax(int dim) const {
    Tensor exp_tensor = this->exp();
    Tensor sum_tensor = exp_tensor.sum(dim, true);
    
    // avoid division by zero
    sum_tensor = sum_tensor.clamp(1e-10, std::numeric_limits<dtype>::max());

    Tensor output = exp_tensor / sum_tensor;
    
    return output;
  }

  Tensor softmax(const Tensor& tensor, int dim) {
    return tensor.softmax(dim);
  }

  /*
    helper constructor
  */

  Tensor Tensor::ones_like() const {
    return ones(shape_);
  }
  Tensor Tensor::zeros_like() const {
    return zeros(shape_);
  }
  Tensor Tensor::randn_like() const {
    return randn(shape_);
  }
  Tensor Tensor::empty_like() const {
    return empty(shape_);
  }


  std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(\n  shape=" << tensor.shape_ << ", strides=" << tensor.stride_ << "\n  data={\n";
    std::string prefix = "    ";
    os << prefix;
    print_tensor_data_recursive(os, tensor, 0, 0, prefix + " ");
    os << "\n  }\n)\n";
    return os;
  }

  Tensor Tensor::permute(veci p) const {
    if ((int)p.size() > dim_)
      throw std::invalid_argument("Tensor: permute size exceeds dim");

    for (int& d: p)
      d = normalize_index(d, dim_);
    
    veci sorted_p = p;
    std::sort(sorted_p.begin(), sorted_p.end());
    auto p_end = std::unique(sorted_p.begin(), sorted_p.end());
    if (p_end != sorted_p.end())
      throw std::invalid_argument("Tensor: permute contains duplicate dimensions");

    shape_t new_shape = shape_;
    stride_t new_stride = stride_;
    for (int i = 0; i < (int)p.size(); i++) {
      new_shape[sorted_p[i]] = shape_[p[i]];
      new_stride[sorted_p[i]] = stride_[p[i]];
    }

    return Tensor(new_shape, new_stride, offset_, storage_);
  }

  Tensor Tensor::transpose(int dim1, int dim2) const {
    dim1 = normalize_index(dim1, dim_);
    dim2 = normalize_index(dim2, dim_);
    if (dim1 == dim2)
      return *this;
    if (dim1 < dim2)
      std::swap(dim1, dim2);
    // dim1 > dim2
    return permute(veci{dim1, dim2});
  }

  Tensor Tensor::reshape(const shape_t& purposed_shape, bool copy) const {
    shape_t new_shape = induce_shape(purposed_shape);

    if (!is_contiguous() || copy) {
      return contiguous().view(new_shape);
    } else {
      return view(new_shape);
    }
  }

  Tensor Tensor::view(const shape_t& purposed_shape) const {
    shape_t new_shape = induce_shape(purposed_shape);

    if (!is_contiguous())
      throw std::runtime_error("Tensor: view on non-contiguous tensor is not allowed");

    int new_numel = calc_numel(new_shape);
    if (new_numel != numel_)
      throw std::invalid_argument("Tensor: view shape numel mismatch");

    stride_t new_stride = calc_stride(new_shape);

    return Tensor(new_shape, new_stride, offset_, storage_);
  }

  Tensor Tensor::narrow(int dim, int start, int length, bool copy) const {
    dim = normalize_index(dim, dim_);
    start = normalize_index(start, shape_[dim]);
    if (length < 0 || start + length > shape_[dim])
      throw std::out_of_range("Tensor: narrow length out of range");

    shape_t new_shape = shape_;
    new_shape[dim] = length;
    int new_offset = offset_ + start * stride_[dim];

    Tensor output = Tensor(new_shape, stride_, new_offset, storage_);
    if (copy)
      return output.clone();
    else
      return output;
  }
  vec<Tensor> Tensor::chunk(int dim, int num_chunk) const {
    if (num_chunk <= 0)
      throw std::invalid_argument("Tensor: number of chunks must be positive");
    int dim_size = size(dim);
    int chunk_size = ceil_div(dim_size, num_chunk);

    vec<Tensor> outputs;
    for (int begin = 0; begin < dim_size; begin += chunk_size) {
      int length = std::min(chunk_size, dim_size - begin);
      outputs.push_back(narrow(dim, begin, length));
      begin += length;
    }
    return outputs;
  }

  vec<Tensor> Tensor::split(int dim, int split_size) const {
    dim = normalize_index(dim, dim_);
    if (split_size <= 0)
      throw std::invalid_argument("Tensor: split size must be positive");
    vec<Tensor> outputs;

    int dim_size = size(dim);
    for (int begin = 0; begin < dim_size; begin += split_size) {
      outputs.push_back(narrow(dim, begin, std::min(split_size, dim_size - begin)));
    }
    return outputs;
  }

  vec<Tensor> Tensor::split(int dim, veci split_sections) const {
    dim = normalize_index(dim, dim_);
    if (std::accumulate(split_sections.begin(), split_sections.end(), 0) != shape_[dim])
      throw std::invalid_argument("Tensor: split sections size mismatch");
  
    vec<Tensor> outputs;
    int begin = 0;
    for (int section : split_sections) {
      if (section <= 0)
        throw std::invalid_argument("Tensor: split section size must be positive");
      outputs.push_back(narrow(dim, begin, section));
      begin += section;
    }
    return outputs;
  }

  Tensor Tensor::stack(const vec<Tensor>& inputs, int dim) {
    if (inputs.empty())
      return Tensor();
    shape_t shape = inputs[0].shape_;
    int num_input = inputs.size();
    for (int i = 1; i < num_input; i++)
      if (!check_shape_match(shape, inputs[i].shape_))
        throw std::invalid_argument("Tensor: stack shape mismatch");
    dim = normalize_index(dim, inputs[0].dim_ + 1);
    
    shape_t new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, num_input);
    vec<slice_t> slices(new_shape.size());
    for (int i = 0; i < (int)new_shape.size(); i++)
      slices[i] = slice_t(0, new_shape[i]);

    Tensor output = Tensor(new_shape);
    for (int i = 0; i < num_input; i++) {
      slices[dim] = slice_t(i, i+1);
      output[slices].squeeze(dim).copy_(inputs[i]);
    }
    return output;
  }

  Tensor Tensor::cat(const vec<Tensor>& inputs, int dim) {
    if (inputs.empty())
      return Tensor();
    
    dim = normalize_index(dim, inputs[0].dim_);
    shape_t shape = inputs[0].shape_;
    shape[dim] = 0;

    int num_input = inputs.size();
    for (int i = 1; i < num_input; i++) {
      shape_t other_shape = inputs[i].shape_;
      if (other_shape.size() != shape.size())
        throw std::invalid_argument("Tensor: cat dimention mismatch");

      other_shape[dim] = 0;
      if (!check_shape_match(shape, other_shape))
        throw std::invalid_argument("Tensor: cat shape mismatch at non-concatenation dimension");
    }
    
    int total_size = 0;
    for (const auto& input : inputs)
      total_size += input.size(dim);
    
    shape_t new_shape = shape;
    new_shape[dim] = total_size;

    Tensor output(new_shape);
    int concat_offset = 0;
    
    vec<slice_t> slices(new_shape.size());
    for (int i = 0; i < (int)new_shape.size(); i++)
      slices[i] = slice_t(0, new_shape[i]);
    
    for (const auto& input : inputs) {
      slices[dim] = slice_t(concat_offset, concat_offset + input.size(dim));
      output[slices].copy_(input);
      concat_offset += input.size(dim);
    }
    
    return output;
  }

  Tensor Tensor::squeeze(int dim) const {
    dim = normalize_index(dim, dim_);
    if (shape_[dim] != 1)
      throw std::invalid_argument("Tensor: squeeze on non-singleton dimension");
    shape_t new_shape = shape_;
    stride_t new_stride = stride_;
    new_shape.erase(new_shape.begin() + dim);
    new_stride.erase(new_stride.begin() + dim);
    return Tensor(new_shape, new_stride, offset_, storage_);
  }

  Tensor Tensor::unsqueeze(int dim) const {
    dim = normalize_index(dim, dim_ + 1);
    shape_t new_shape = shape_;
    stride_t new_stride = stride_;
    new_shape.insert(new_shape.begin() + dim, 1);
    if (dim == dim_)
      new_stride.insert(new_stride.begin() + dim, 1);
    else
      new_stride.insert(new_stride.begin() + dim, stride_[dim] * shape_[dim]);
    return Tensor(new_shape, new_stride, offset_, storage_);
  }

  Tensor Tensor::broadcast_to(const shape_t& shape) const {
    shape_t new_shape = broadcast_shape(shape_, shape);
    if (check_shape_match(new_shape, shape_))
      return *this;
    int new_dim = new_shape.size();
    stride_t new_stride = concat_vec(stride_t(new_dim - dim_, 0), stride_);
    for (int i = 0; i < dim_; i++)
      if (shape_[dim_ - 1 - i] == 1 && new_shape[new_dim - 1 - i] != 1)
        new_stride[new_dim - 1 - i] = 0;
    return Tensor(new_shape, new_stride, offset_, storage_);
  }

  std::pair<Tensor, Tensor> Tensor::broadcast(const Tensor& lhs, const Tensor& rhs) {
    shape_t new_shape = broadcast_shape(lhs.shape_, rhs.shape_);
    Tensor lhs_b = lhs.broadcast_to(new_shape);
    Tensor rhs_b = rhs.broadcast_to(new_shape);
    return std::make_pair(lhs_b, rhs_b);
  }
  vec<Tensor> Tensor::broadcast(const vec<Tensor>& tensors) {
    if (tensors.empty())
      return vec<Tensor>();

    shape_t new_shape = tensors[0].shape_;
    for (int i = 1; i < (int)tensors.size(); i++) {
      new_shape = broadcast_shape(new_shape, tensors[i].shape_);
    }

    vec<Tensor> outputs;
    for (const auto& tensor : tensors) {
      outputs.push_back(tensor.broadcast_to(new_shape));
    }
    return outputs;
  }



  /*
    helper constructors
  */
  Tensor to_singleton_tensor(dtype value, int dim) {
    if (dim < 0)
      throw std::invalid_argument("to_singleton_tensor: dim must be non-negative");
    
    return Tensor(shape_t(dim, 1), value);
  }

  Tensor ones(const shape_t& shape) {
    if (shape.empty())
      throw std::invalid_argument("ones: shape cannot be empty");
    return Tensor(shape, 1.0);
  }
  Tensor ones_like(const Tensor& ref) {
    return ones(ref.size());
  }

  Tensor zeros(const shape_t& shape) {
    if (shape.empty())
      throw std::invalid_argument("zeros: shape cannot be empty");
    return Tensor(shape, 0.0);
  }
  Tensor zeros_like(const Tensor& ref) {
    return zeros(ref.size());
  }

  Tensor randn(const shape_t& shape) {
    if (shape.empty())
      throw std::invalid_argument("randn: shape cannot be empty");
    
    std::function<dtype()> randn = []() -> dtype {
      return random::randn(random::mt19937_rng, 0, 1);
    };
    
    Tensor output = Tensor(shape, randn);
    return output;
  }
  Tensor randn_like(const Tensor& ref) {
    return ref.randn_like();
  }

  Tensor empty(const shape_t& shape) {
    // if shape is empty, a scalar is created instead of pure empty tensor
    return Tensor(shape);
  }
  Tensor empty_like(const Tensor& ref) {
    return ref.empty_like();
  }

  Tensor arange(dtype start, dtype end, dtype step) {
    if (step <= 0)
      throw std::invalid_argument("arange: step must be positive");
    if (start >= end)
      return Tensor();
    int numel = ceil_div(end - start, step);
    Tensor output = Tensor(shape_t({numel}));
    for (int i = 0; i < numel; i++)
      output[i] = start + i * step;
    return output;
  }

  Tensor range(dtype start, dtype end, dtype step) {
    if (step <= 0)
      throw std::invalid_argument("range: step must be positive");
    if (start >= end)
      return Tensor();
    int numel = (end - start) / step + 1;
    Tensor output = Tensor(shape_t({numel}));
    for (int i = 0; i < numel; i++)
      output[i] = start + i * step;
    return output;
  }

  Tensor linspace(dtype start, dtype end, int num_steps) {
    if (num_steps <= 0)
      throw std::invalid_argument("linspace: num_steps must be positive");

    Tensor output = Tensor(shape_t({num_steps}));
    dtype step = (end - start) / (num_steps - 1);
    for (int i = 0; i < num_steps; i++)
      output[i] = start + i * step;
    
    return output;
  }

};