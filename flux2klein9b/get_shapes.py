#!/usr/bin/env python3
"""
Diagnostic script to see actual transformer input shapes
"""

import torch
from diffusers import DiffusionPipeline

class TransformerSpy:
    """Wrapper that prints all inputs then calls the real transformer"""
    
    def __init__(self, real_transformer):
        self.real_transformer = real_transformer
        self.call_count = 0
    
    def __call__(self, *args, **kwargs):
        self.call_count += 1
        
        print(f"\n{'='*60}")
        print(f"Transformer call #{self.call_count}")
        print(f"{'='*60}")
        
        print("\nPositional args:")
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                print(f"  args[{i}]: shape={arg.shape}, dtype={arg.dtype}, device={arg.device}")
            else:
                print(f"  args[{i}]: {type(arg)} = {arg}")
        
        print("\nKeyword args:")
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
            else:
                print(f"  {key}: {type(val)} = {val}")
        
        # Call the real transformer
        result = self.real_transformer(*args, **kwargs)
        
        print("\nOutput:")
        if hasattr(result, 'sample'):
            print(f"  result.sample: shape={result.sample.shape}, dtype={result.sample.dtype}")
        elif isinstance(result, tuple):
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    print(f"  result[{i}]: shape={r.shape}, dtype={r.dtype}")
        
        return result
    
    def __getattr__(self, name):
        """Forward any other attributes to the real transformer"""
        return getattr(self.real_transformer, name)


def main():
    print("="*60)
    print("FLUX.2-klein-9B Diagnostic - See Actual Shapes")
    print("="*60)
    
    print("\nLoading pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    print("✓ Pipeline loaded")
    
    # Wrap the transformer with our spy
    print("\nWrapping transformer with diagnostic spy...")
    pipe.transformer = TransformerSpy(pipe.transformer)
    print("✓ Transformer wrapped")
    
    # Generate a small image to see the shapes
    print("\nGenerating 512x512 image (faster for diagnostics)...")
    print("This will print the actual shapes the transformer receives!\n")
    
    image = pipe(
        prompt="a cat",
        height=512,
        width=512,
        num_inference_steps=4,  # Just 4 steps to see the pattern quickly
        guidance_scale=3.5,
    ).images[0]
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nUse the shapes printed above to create your ONNX export!")
    print("The transformer was called", pipe.transformer.call_count, "times")
    
    image.save("diagnostic_output.png")
    print(f"\nTest image saved to diagnostic_output.png")


if __name__ == "__main__":
    main()
