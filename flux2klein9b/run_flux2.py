#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import tensorrt as trt
from pathlib import Path
from diffusers import DiffusionPipeline
from torch.profiler import profile, ProfilerActivity


ONNX_PATH = "onnx1/flux_transformer_opset18.onnx"
ENGINE_PATH = "onnx1/flux_transformer.trt"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class RMSNormONNX(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def replace_rms_norm(model):
    """Replace all RMSNorm layers with ONNX-compatible version"""
    print("\nReplacing RMSNorm layers...")
    
    count = 0
    
    def recursive_replace(module, prefix=''):
        nonlocal count
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            class_name = child.__class__.__name__
            
            if 'RMSNorm' in class_name or 'RmsNorm' in class_name:
                if hasattr(child, 'weight'):
                    weight = child.weight.data
                    eps = getattr(child, 'eps', getattr(child, 'variance_epsilon', 1e-6))
                    setattr(module, name, RMSNormONNX(weight, eps))
                    count += 1
            else:
                recursive_replace(child, full_name)
    
    recursive_replace(model)
    print(f"Replaced {count} RMSNorm layers")
    return model


def export_to_onnx(pipe):
    """Export transformer to ONNX"""
    print("\n" + "="*70)
    print("STEP 1: Export to ONNX")
    print("="*70)
    
    if Path(ONNX_PATH).exists():
        print(f"ONNX file already exists: {ONNX_PATH}")
        return True
    
    transformer = pipe.transformer.eval().cuda()
    transformer = replace_rms_norm(transformer)
    
    # Prepare inputs
    batch = 1
    num_img_tokens = 4096
    num_txt_tokens = 512
    
    hidden_states = torch.randn(batch, num_img_tokens, 128, dtype=torch.float16, device="cuda")
    encoder_hidden_states = torch.randn(batch, num_txt_tokens, 12288, dtype=torch.float16, device="cuda")
    timestep = torch.randn(batch, dtype=torch.float16, device="cuda")
    txt_ids = torch.randint(0, 100, (batch, num_txt_tokens, 4), dtype=torch.int64, device="cuda")
    img_ids = torch.randint(0, 100, (batch, num_img_tokens, 4), dtype=torch.int64, device="cuda")

    # Test forward
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=None,
            return_dict=False
        )
        print(f"Forward pass works! Output: {output[0].shape}")
    
    # Wrapper
    class Wrapper(nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer
        
        def forward(self, h, e, t, txt, img):
            return self.transformer(
                hidden_states=h,
                encoder_hidden_states=e,
                timestep=t,
                txt_ids=txt,
                img_ids=img,
                guidance=None,
                return_dict=False
            )[0]
    
    wrapped = Wrapper(transformer)
    
    # Export
    print("\nExporting to ONNX (this takes 10-30 minutes)...")
    print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
    
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (hidden_states, encoder_hidden_states, timestep, txt_ids, img_ids),
            ONNX_PATH,
            opset_version=18,
            input_names=['hidden_states', 'encoder_hidden_states', 'timestep', 'txt_ids', 'img_ids'],
            output_names=['output'],
            do_constant_folding=True,
            dynamo=False,
        )
    
    import os
    size_mb = os.path.getsize(ONNX_PATH) / (1024*1024)
    print(f"ONNX exported: {ONNX_PATH} ({size_mb:.1f} MB)")
    print(f"Completed: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
    return True


def build_trt_engine():
    """Build TensorRT engine from ONNX"""
    print("\n" + "="*70)
    print("STEP 2: Build TensorRT Engine")
    print("="*70)
    
    if Path(ENGINE_PATH).exists():
        print(f"TensorRT engine already exists: {ENGINE_PATH}")
        return True
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print(f"\nParsing {ONNX_PATH}...")
    with open(ONNX_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            print("✗ Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return False
    
    # Configure
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8GB
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Create optimization profile for dynamic shapes
    print("\nSetting up optimization profile for dynamic shapes...")
    profile = builder.create_optimization_profile()
    
    # Define shapes: min, opt(imal), max
    # For 512x512: img_tokens=1024, txt_tokens=512
    # For 1024x1024: img_tokens=4096, txt_tokens=512
    
    # hidden_states: [batch, img_tokens, 128]
    profile.set_shape('hidden_states', 
                      min=(1, 4096, 128),    # 512x512
                      opt=(1, 4096, 128),    # 512x512 (optimize for this)
                      max=(1, 4096, 128))    # 1024x1024
    
    # encoder_hidden_states: [batch, txt_tokens, 12288]
    temp = (1, 512, 12288)
    profile.set_shape('encoder_hidden_states', min=temp, opt=temp, max=temp)  # Fixed size
    
    # timestep: [batch]
    profile.set_shape('timestep', min=(1,), opt=(1,), max=(1,))
    
    # txt_ids: [batch, txt_tokens, 4]
    temp = (1, 512, 4)
    profile.set_shape('txt_ids', min=temp, opt=temp, max=temp)  # Fixed size
    
    # img_ids: [batch, img_tokens, 4]
    profile.set_shape('img_ids', 
                      min=(1, 4096, 4),      # 512x512
                      opt=(1, 4096, 4),      # 512x512
                      max=(1, 4096, 4))      # 1024x1024
    
    config.add_optimization_profile(profile)
    print("Optimization profile configured")
    print("  Optimized for 512x512 images (can handle up to 1024x1024)")
    
    print("\nBuilding TensorRT engine (10-30 minutes)...")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if not serialized_engine:
        print("✗ Failed to build engine")
        return False
    
    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized_engine)
    
    import os
    size_mb = os.path.getsize(ENGINE_PATH) / (1024*1024)
    print(f"TensorRT engine built: {ENGINE_PATH} ({size_mb:.1f} MB)")
    print(f"Completed: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
    return True


class TRTTransformer:
    """TensorRT transformer wrapper that mimics original transformer interface"""
    
    def __init__(self, engine_path, original_transformer=None):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # Preserve attributes from original transformer
        if original_transformer is not None:
            self.config = original_transformer.config
            self.dtype = original_transformer.dtype
            self.device = original_transformer.device
        else:
            # Fallback: create minimal attributes
            from types import SimpleNamespace
            self.config = SimpleNamespace(in_channels=64)
            self.dtype = torch.float16
            self.device = torch.device('cuda:0')
    
    def to(self, *args, **kwargs):
        """Dummy to() method - TRT engine is already on GPU"""
        return self
    
    def parameters(self):
        """Dummy parameters() - TRT has no trainable parameters"""
        return []
    
    def eval(self):
        """Dummy eval() - TRT is always in eval mode"""
        return self
    
    def cache_context(self, *args, **kwargs):
        """Dummy cache_context - return a no-op context manager"""
        from contextlib import nullcontext
        return nullcontext()
    
    def __call__(self, hidden_states, encoder_hidden_states=None, timestep=None,
                 txt_ids=None, img_ids=None, guidance=None, return_dict=True, **kwargs):
        
        # Convert to contiguous tensors
        hidden_states = hidden_states.contiguous().half().cuda()
        encoder_hidden_states = encoder_hidden_states.contiguous().half().cuda()
        timestep = timestep.contiguous().half().cuda()
        txt_ids = txt_ids.contiguous().long().cuda()
        img_ids = img_ids.contiguous().long().cuda()
        
        # Set shapes
        self.context.set_input_shape('hidden_states', tuple(hidden_states.shape))
        self.context.set_input_shape('encoder_hidden_states', tuple(encoder_hidden_states.shape))
        self.context.set_input_shape('timestep', tuple(timestep.shape))
        self.context.set_input_shape('txt_ids', tuple(txt_ids.shape))
        self.context.set_input_shape('img_ids', tuple(img_ids.shape))
        
        # Set addresses
        self.context.set_tensor_address('hidden_states', hidden_states.data_ptr())
        self.context.set_tensor_address('encoder_hidden_states', encoder_hidden_states.data_ptr())
        self.context.set_tensor_address('timestep', timestep.data_ptr())
        self.context.set_tensor_address('txt_ids', txt_ids.data_ptr())
        self.context.set_tensor_address('img_ids', img_ids.data_ptr())
        
        # Output
        output_shape = tuple(self.context.get_tensor_shape('output'))
        output = torch.empty(output_shape, dtype=torch.float16, device='cuda')
        self.context.set_tensor_address('output', output.data_ptr())
        
        # Execute on current stream (async)
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)

        if return_dict:
            return type('Output', (), {'sample': output})()
        return (output,)
    
    def __getattr__(self, name):
        """Catch-all for any other methods - return no-op context manager or self"""
        if name.endswith('_context'):
            from contextlib import nullcontext
            return lambda *args, **kwargs: nullcontext()
        # For other methods, return self to allow chaining
        return lambda *args, **kwargs: self


def run_with_profiler(
    pipe,
    *,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    steps: int = 4,
    warmup: int = 3,
    active: int = 1,
    out_png: str = "flux_trt_output.png",
    trace_dir: str = "./traces",
):
    # Warmup (no profiler)
    if warmup > 0:
        print(f"[PROF] Warmup: {warmup} run(s)")
        for i in range(warmup):
            _ = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=1.0,
            ).images[0]
        torch.cuda.synchronize()

    print(f"[PROF] Profiling: {active} run(s) -> trace_dir={trace_dir}")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # set True if you want Python stack traces (overhead)
        with_flops=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        # Use a schedule so you can capture exactly 'active' iterations
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=active, repeat=1),
    ) as prof:
        last_img = None
        for i in range(active):
            last_img = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=1.0,
            ).images[0]
            torch.cuda.synchronize()
            prof.step()

    if last_img is not None:
        last_img.save(out_png)
        print(f"[PROF] Saved image: {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trt",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Enable TensorRT (true/false)"
    )
    args = parser.parse_args()

    print("="*70)
    print("FLUX.2-klein-9B with TensorRT")
    print("="*70)
    
    # Load pipeline
    print("\nLoading pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    print("Pipeline loaded")
    
    # Keep reference to original transformer for config
    original_transformer = pipe.transformer
   
    if args.trt:
        # Export to ONNX (skip if already exists)
        if not Path(ONNX_PATH).exists():
            print("\nONNX file not found, exporting...")
            if not export_to_onnx(pipe):
                return
        else:
            print(f"\nUsing existing ONNX file: {ONNX_PATH}")
    
        # Build TRT engine (skip if already exists)
        if not Path(ENGINE_PATH).exists():
            print("\nTensorRT engine not found, building...")
            if not build_trt_engine():
                return
        else:
            print(f"\nUsing existing TensorRT engine: {ENGINE_PATH}")
    
        # Load TRT engine
        print("\n" + "="*70)
        print("Loading TensorRT Engine")
        print("="*70)
        pipe.transformer = TRTTransformer(ENGINE_PATH, original_transformer)
        print("Transformer replaced with TensorRT")
    
    print("\n" + "="*70)
    run_with_profiler(pipe, prompt="a serene mountain lake at sunset, highly detailed")

if __name__ == "__main__":
    main()
