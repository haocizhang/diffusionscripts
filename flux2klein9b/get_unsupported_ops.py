#!/usr/bin/env python3
"""
Identify all PyTorch operators in FLUX.2-klein-9B that don't have ONNX support
"""

import torch
from diffusers import DiffusionPipeline
import io
from contextlib import redirect_stdout, redirect_stderr
import re


def capture_onnx_errors(model, inputs):
    """Try to export and capture all ONNX errors"""
    
    errors = []
    
    # Try export with verbose to catch all errors
    try:
        # Redirect output to capture error messages
        f = io.StringIO()
        
        with redirect_stdout(f), redirect_stderr(f):
            torch.onnx.export(
                model,
                inputs,
                "temp.onnx",
                opset_version=17,
                verbose=True,
                dynamo=False,
            )
    except Exception as e:
        error_text = str(e)
        
        # Extract operator names from error messages
        # Pattern 1: 'aten::operator_name'
        operators = re.findall(r"'aten::([^']+)'", error_text)
        
        # Pattern 2: OpOverload(op='aten.operator_name'
        operators.extend(re.findall(r"op='aten\.([^']+)'", error_text))
        
        # Pattern 3: torch.ops.aten.operator_name
        operators.extend(re.findall(r"torch\.ops\.aten\.([^\s\)]+)", error_text))
        
        errors.append({
            'error': str(e)[:500],
            'operators': list(set(operators))
        })
    
    return errors


def analyze_model_operators(pipe):
    """Analyze which operators are used in the model"""
    print("="*70)
    print("ANALYZING FLUX.2-klein-9B ONNX COMPATIBILITY")
    print("="*70)
    
    transformer = pipe.transformer.eval().cuda()
    
    # Prepare test inputs
    batch = 1
    num_img_tokens = 1024
    num_txt_tokens = 512
    
    hidden_states = torch.randn(batch, num_img_tokens, 128, dtype=torch.float16, device="cuda")
    encoder_hidden_states = torch.randn(batch, num_txt_tokens, 12288, dtype=torch.float16, device="cuda")
    timestep = torch.randn(batch, dtype=torch.float16, device="cuda")
    txt_ids = torch.randint(0, 100, (batch, num_txt_tokens, 4), dtype=torch.int64, device="cuda")
    img_ids = torch.randint(0, 100, (batch, num_img_tokens, 4), dtype=torch.int64, device="cuda")
    
    # Wrapper for export
    class TransformerWrapper(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer
        
        def forward(self, hidden_states, encoder_hidden_states, timestep, txt_ids, img_ids):
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                txt_ids=txt_ids,
                img_ids=img_ids,
                guidance=None,
                return_dict=False
            )[0]
    
    wrapped = TransformerWrapper(transformer)
    
    print("\nAttempting ONNX export to identify unsupported operators...")
    print("(This will fail - we're capturing the errors)\n")
    
    # Try different opsets to see which operators fail at each level
    results = {}
    
    for opset in [13, 14, 15, 16, 17, 18]:
        print(f"\n{'='*70}")
        print(f"Testing OPSET {opset}")
        print('='*70)
        
        try:
            torch.onnx.export(
                wrapped,
                (hidden_states, encoder_hidden_states, timestep, txt_ids, img_ids),
                f"temp_opset{opset}.onnx",
                opset_version=opset,
                verbose=False,
                dynamo=False,
            )
            print(f"✓ OPSET {opset} SUCCEEDED!")
            results[opset] = {"status": "success", "unsupported_ops": []}
            break
            
        except Exception as e:
            error_str = str(e)
            
            # Extract unsupported operators
            unsupported_ops = set()
            
            # Pattern: operator 'aten::op_name'
            matches = re.findall(r"'aten::([^']+)'", error_str)
            unsupported_ops.update(matches)
            
            # Pattern: OpOverload(op='aten.op_name'
            matches = re.findall(r"op='aten\.([^']+)'", error_str)
            unsupported_ops.update(matches)
            
            # Pattern: torch.ops.aten.op_name
            matches = re.findall(r"torch\.ops\.aten\.([a-zA-Z_0-9]+)", error_str)
            unsupported_ops.update(matches)
            
            # Get the main error message
            error_lines = error_str.split('\n')
            main_error = error_lines[0] if error_lines else error_str[:200]
            
            results[opset] = {
                "status": "failed",
                "unsupported_ops": list(unsupported_ops),
                "error": main_error[:300]
            }
            
            print(f"✗ OPSET {opset} FAILED")
            if unsupported_ops:
                print(f"\nUnsupported operators found:")
                for op in sorted(unsupported_ops):
                    print(f"  • aten::{op}")
            
            print(f"\nMain error: {main_error}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF UNSUPPORTED OPERATORS")
    print("="*70)
    
    all_unsupported = set()
    for opset, result in results.items():
        if result['status'] == 'failed':
            all_unsupported.update(result['unsupported_ops'])
    
    if all_unsupported:
        print(f"\nFound {len(all_unsupported)} unique unsupported operators:\n")
        for i, op in enumerate(sorted(all_unsupported), 1):
            print(f"{i:2d}. aten::{op}")
        
        print("\n" + "="*70)
        print("DIFFICULTY ASSESSMENT")
        print("="*70)
        
        critical_ops = {
            '_fused_rms_norm': 'Fused operation - can be decomposed',
            'scaled_dot_product_attention': 'Attention - supported in opset >= 14',
            '_scaled_dot_product_flash_attention': 'Flash attention - no ONNX equivalent',
            '_scaled_dot_product_efficient_attention': 'Memory efficient attention - no ONNX equivalent',
        }
        
        found_critical = []
        for op in all_unsupported:
            if op in critical_ops:
                found_critical.append((op, critical_ops[op]))
        
        if found_critical:
            print("\nCritical operators found:")
            for op, desc in found_critical:
                print(f"  • {op}: {desc}")
        
        print(f"\nTotal operators to patch/work around: {len(all_unsupported)}")
        
        if len(all_unsupported) <= 3:
            print("\n✓ FEASIBLE: Only a few operators to patch")
        elif len(all_unsupported) <= 10:
            print("\n⚠ MODERATE: Significant work but possible")
        else:
            print("\n✗ DIFFICULT: Too many operators to reasonably patch")
            print("   Recommendation: Use torch.compile or a different model")
    else:
        print("\n✓ All operators are supported! Export should work.")
    
    print("\n" + "="*70)


def main():
    print("\nLoading FLUX.2-klein-9B pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    print("✓ Pipeline loaded\n")
    
    analyze_model_operators(pipe)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
