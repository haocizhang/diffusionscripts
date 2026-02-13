import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# -------------------------
# 1. Define Model
# -------------------------

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(16, 32, bias=False)
        self.l2 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.l2(torch.relu(self.l(x)))

torch.manual_seed(0)
model = Tiny().eval().cuda()

# -------------------------
# 2. Export ONNX
# -------------------------

dummy = torch.randn(1, 16, device="cuda")

torch.onnx.export(
    model,
    dummy,
    "tiny.onnx",
    input_names=["x"],
    output_names=["y"],
    opset_version=18,
    dynamic_axes={"x": {0: "B"}, "y": {0: "B"}},
)

print("Exported ONNX.")

# -------------------------
# 3. Build TensorRT Engine (TRT 10.x API)
# -------------------------

def build_engine(onnx_path):
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 16), (8, 16), (64, 16))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized)

engine = build_engine("tiny.onnx")
context = engine.create_execution_context()

print("Built TensorRT engine.")

# -------------------------
# 4. Inference Comparison
# -------------------------

def run_trt(x_np):
    stream = cuda.Stream()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    context.set_input_shape(input_name, x_np.shape)

    out_shape = tuple(context.get_tensor_shape(output_name))
    y_np = np.empty(out_shape, dtype=np.float32)

    d_in = cuda.mem_alloc(x_np.nbytes)
    d_out = cuda.mem_alloc(y_np.nbytes)

    context.set_tensor_address(input_name, int(d_in))
    context.set_tensor_address(output_name, int(d_out))

    cuda.memcpy_htod_async(d_in, x_np, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(y_np, d_out, stream)
    stream.synchronize()

    return y_np

# Test input
x_test = torch.randn(4, 16, device="cuda")
with torch.no_grad():
    y_torch = model(x_test).cpu().numpy()

y_trt = run_trt(x_test.cpu().numpy())

print("Max abs diff:", np.max(np.abs(y_torch - y_trt)))

