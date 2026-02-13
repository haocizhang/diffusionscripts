# 1) Setup

Make sure you finish the setup procedure in https://github.com/haocizhang/diffusionscripts/tree/main and you are running in your Python env: `source .venv_trt/bin/activate`

# 2) Running Flux2 Klein 9B w/ or w/o TRT & ONNX
```
python run_flux2.py --trt true
python run_flux2.py --trt false
```

# 3) Understanding which ops are not supported by ONNX
```
python get_unsupported_ops.py
```

# 4) Run a dummy prompt and get the input shapes to the transformer
(This would be useful if you want to trace the model using onnx)
```
python get_shapes.py
```
