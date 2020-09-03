#include <torch/extension.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <sstream>
#include <NvInfer.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdexcept>

using namespace nvinfer1;

namespace torch2trt {

__device__ __forceinline__ int idx(const int n, const int num_channels,
                                   const int c, const int height,
                                   const int width, const int y, const int x) {
  return ((n * num_channels + c) * height + y) * width + x;
}

__device__ __forceinline__ float area_pixel_compute_source_index(
    const float scale,
    const int dst_index,
    const bool align_corners) {
	if (align_corners) {
		return scale * dst_index;
	} else {
		const float src_idx = scale * (dst_index + 0.5) - 0.5;
		return (src_idx < 0) ? 0.0f : src_idx;
	}
//  return align_corners ? scale * dst_index : scale * (dst_index + 0.5f) - 0.5f;
}


// input is X, output is Y
template <typename scalar_t>
__global__ void bilinearForwardKernel(
    const int batch_size,
    const int output_size, const int num_channels, const int input_height,
    const int input_width, const int output_height, const int output_width,
    const scalar_t *const __restrict__ X, scalar_t *const __restrict__ Y, const bool align_corners) {

  const int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index >= output_size) {
	  return;
  }

  int indexTemp = index;
  const int w2 = indexTemp % output_width;
  indexTemp /= output_width;
  const int h2 = indexTemp % output_height;

//  const int w2 = index % output_width;
//  const int h2 = index / output_width;

  float rheight, rwidth;
  if (align_corners) {
      rheight = output_height > 1 ? 1.f * (input_height - 1) / (output_height - 1) : 0.f;
      rwidth = output_width > 1 ? 1.f * (input_width - 1) / (output_width - 1) : 0.f;
  } else {
      rheight = output_height > 0 ? (1.f * input_height / output_height) : 0.f;
      rwidth = output_width > 0 ? (1.f * input_width / output_width) : 0.f;
  }

  const float h1r = area_pixel_compute_source_index(rheight, h2, align_corners); //  rheight * out_y;
  const int h1 = static_cast<int>(h1r);
  const int h1p = (h1 < input_height - 1) ? 1 : 0;
  const float h1lambda = h1r - h1;
  const float h0lambda = 1.f - h1lambda;

  const float w1r = area_pixel_compute_source_index(rwidth, w2, align_corners); //rwidth * out_x;
  const int w1 = static_cast<int>(w1r);
  const int w1p = (w1 < input_width - 1) ? 1 : 0;
  const float w1lambda = w1r - w1;
  const float w0lambda = 1.f - w1lambda;

 for (int n = 0; n < batch_size; n++) {
    for (int c = 0; c < num_channels; c++) {
      Y[idx(n, num_channels, c, output_height, output_width, h2, w2)] =
          static_cast<scalar_t>(
              h0lambda *
                  (w0lambda * static_cast<float>(__ldg(&X[idx(n, num_channels, c, input_height,
                                           input_width, h1, w1)])) +
                   w1lambda * static_cast<float>(__ldg(&X[idx(n, num_channels, c, input_height,
                                           input_width, h1, w1 + w1p)]))) +
              h1lambda *
                  (w0lambda * static_cast<float>(__ldg(&X[idx(n, num_channels, c, input_height,
                                           input_width, h1 + h1p, w1)])) +
                   w1lambda * static_cast<float>(__ldg(&X[idx(n, num_channels, c, input_height,
                                           input_width, h1 + h1p, w1 + w1p)]))));
    }
  }
}

class InterpolatePlugin : public IPluginV2 {
private:
    
  // configured by class
  at::TensorOptions tensor_options;
  std::vector<int64_t> input_sizes;
  std::vector<int64_t> output_sizes;
  DataType dtype;
    
  // configured by user
  std::vector<int64_t> size;
  std::string mode;
  bool align_corners;

public:
    
  // create from arguments
  InterpolatePlugin(std::vector<int64_t> size, std::string mode, bool align_corners) :
    size(size), mode(mode), align_corners(align_corners)
  {}
    
  InterpolatePlugin(const char *data, size_t length) : InterpolatePlugin(std::string(data, length)) {}
    
  // create from serialized data
  InterpolatePlugin(const std::string &data) {
      deserializeFromString(data);
  }
   
  void deserializeFromString(const std::string &data) {
      std::istringstream data_stream(data);
      torch::serialize::InputArchive input_archive;
      input_archive.load_from(data_stream);
      {
          torch::IValue value;
          input_archive.read("size", value);
#ifdef USE_DEPRECATED_INTLIST
          size = value.toIntListRef().vec();
#else
          size = value.toIntVector();
#endif
      }
      {
          torch::IValue value;
          input_archive.read("mode", value);
          mode = value.toStringRef();
      }
      {
          torch::IValue value;
          input_archive.read("align_corners", value);
          align_corners = value.toBool();
      }
      {
          torch::IValue value;
          input_archive.read("dtype", value);
          dtype = (DataType) value.toInt();
      }
      {
          torch::IValue value;
          input_archive.read("input_sizes", value);
#ifdef USE_DEPRECATED_INTLIST
          input_sizes = value.toIntListRef().vec();
#else
          input_sizes = value.toIntVector();
#endif
      }
      {
          torch::IValue value;
          input_archive.read("output_sizes", value);
#ifdef USE_DEPRECATED_INTLIST
          output_sizes = value.toIntListRef().vec();
#else
          output_sizes = value.toIntVector();
#endif
      }
  }
    
  std::string serializeToString() const {
      torch::serialize::OutputArchive output_archive;
      output_archive.write("size", torch::IValue(size));
      output_archive.write("mode", torch::IValue(mode));
      output_archive.write("align_corners", torch::IValue(align_corners));
      output_archive.write("dtype", torch::IValue((int) dtype));
      output_archive.write("input_sizes", torch::IValue(input_sizes));
      output_archive.write("output_sizes", torch::IValue(output_sizes));
      std::ostringstream data_str;
      output_archive.save_to(data_str);
      return data_str.str();
  }

  const char* getPluginType() const override {
    return "interpolate";
  };

  const char* getPluginVersion() const override {
    return "1";
  }

  int getNbOutputs() const override {
    return 1;
  } 

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
    Dims dims;
    dims.nbDims = inputs->nbDims;

    dims.d[0] = inputs->d[0];
    for (int i = 0; i < size.size(); i++) {
      dims.d[i + 1] = size[i];
    }

    return dims;
  }

  bool supportsFormat(DataType type, PluginFormat format) const override {
    if (format != PluginFormat::kNCHW) {
      return false;
    }
    if (type == DataType::kINT32 || type == DataType::kINT8) {
      return false;
    }
    return true;
  }

  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
      int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {
    
    // set data type
    if (type == DataType::kFLOAT) {
      tensor_options = tensor_options.dtype(c10::kFloat);
      dtype = type;
    } else if (type == DataType::kHALF) {
      tensor_options = tensor_options.dtype(c10::kHalf);
      dtype = type;
    }
      
    // set input sizes
    input_sizes.resize(inputDims[0].nbDims);
    for (int i = 0; i < inputDims[0].nbDims; i++) {
      input_sizes[i] = inputDims[0].d[i];
    }

    // set output sizes
    output_sizes.resize(outputDims[0].nbDims);
    for (int i = 0; i < outputDims[0].nbDims; i++) {
      output_sizes[i] = outputDims[0].d[i];
    }
  }

  int initialize() override {
    // set device
    tensor_options = tensor_options.device(c10::kCUDA);
      
    // set data type
    if (dtype == DataType::kFLOAT) {
        tensor_options = tensor_options.dtype(c10::kFloat);
    } else if (dtype == DataType::kHALF) {
        tensor_options = tensor_options.dtype(c10::kHalf);
    }
      
    return 0;
  }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
    // get input / output dimensions
    std::vector<long> batch_input_sizes = input_sizes;
    std::vector<long> batch_output_sizes = output_sizes;
    batch_input_sizes.insert(batch_input_sizes.begin(), batchSize);
    batch_output_sizes.insert(batch_output_sizes.begin(), batchSize);

    const int output_size = static_cast<int>(std::accumulate(batch_output_sizes.begin(), batch_output_sizes.end(), 1, std::multiplies<long>()));
    const int channels = static_cast<int>(input_sizes[0]);
    const int input_height = static_cast<int>(input_sizes[1]);
    const int input_width = static_cast<int>(input_sizes[2]);

    const int output_height = static_cast<int>(output_sizes[1]);
    const int output_width = static_cast<int>(output_sizes[2]);

    dim3 block(32, 1, 1);
    dim3 grid((output_size + block.x - 1) / block.x, 1, 1);

    if (tensor_options.dtype() == torch::kFloat32) {
        bilinearForwardKernel<float><<<grid, block, 0, stream>>>(
		    batchSize,
    		    output_size, channels, input_height, input_width, 
    		    output_height, output_width,
    		    static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), align_corners
	);
    } else if (tensor_options.dtype() == torch::kFloat16) {
        bilinearForwardKernel<__half><<<grid, block, 0, stream>>>(
		    batchSize,
    		    output_size, channels, input_height, input_width, 
    		    output_height, output_width,
    		    static_cast<const __half*>(inputs[0]), static_cast<__half*>(outputs[0]), align_corners
        );
    } else {
	    throw std::runtime_error("interpolation plugin can only operate on fp16 or fp32");
    }

    return 0;
  }

  size_t getSerializationSize() const override {
    return serializeToString().size();
  }
    
  void serialize(void* buffer) const override {
      std::string data = serializeToString();
      size_t size = getSerializationSize();
      data.copy((char *) buffer, size);
  }

  void destroy() override {}

  IPluginV2* clone() const override {
    return new InterpolatePlugin(size, mode, align_corners);
  }

  void setPluginNamespace(const char* pluginNamespace) override {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }

};

class InterpolatePluginCreator : public IPluginCreator {
public:
  InterpolatePluginCreator() {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }

  const char *getPluginName() const override {
    return "interpolate";
  }

  const char *getPluginVersion() const override {
    return "1";
  }

  IPluginV2 *deserializePlugin(const char *name, const void *data, size_t length) override {
    return new InterpolatePlugin((const char*) data, length);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }

};


REGISTER_TENSORRT_PLUGIN(InterpolatePluginCreator);
    

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<InterpolatePlugin>(m, "InterpolatePlugin")
        .def(py::init<std::vector<int64_t>, std::string, bool>(), py::arg("size"), py::arg("mode"), py::arg("align_corners"))
        .def(py::init<const std::string &>(), py::arg("data"))
        .def("getSerializationSize", &InterpolatePlugin::getSerializationSize)
        .def("deserializeFromString", &InterpolatePlugin::deserializeFromString)
        .def("serializeToString", [](const InterpolatePlugin& plugin) {
            std::string data = plugin.serializeToString();
            return py::bytes(data);
        });
}
    
} // namespace torch2trt
