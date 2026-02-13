# 0) Prereqs
You have SSH access to the devserver


Python 3.12 is available (the wheel below is cp312) When reserving the lambda machine, you can reserve the latest package and it will be using 3.12.


You have enough disk space for the tarball + extracted folder



# 1) Download TensorRT tarball locally and copy to devserver
Download:

TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-12.9.tar.gz (I am using the TensorRT 10.15.1 GA for Linux x86_64 and CUDA 12.0 to 12.9 TAR Package)


From: https://developer.nvidia.com/tensorrt/download/10x


Copy to devserver:

```
scp TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-12.9.tar.gz ubuntu@209.20.157.213:~/machgenai
```

# 2) SSH into devserver and extract
```
ssh ubuntu@209.20.157.213
cd ~/machgenai
tar -xvzf TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-12.9.tar.gz
```

This should create:
```
~/machgenai/TensorRT-10.15.1.29/
```


# 3) Set TRT_ROOT (current shell) and verify package layout
```
export TRT_ROOT=$HOME/machgenai/TensorRT-10.15.1.29
ls $TRT_ROOT/python
```

# 4) Create/activate your Python environment (if using venv)
If you already have .venv_trt, activate it; otherwise create it:
```
python -m venv .venv_trt
source .venv_trt/bin/activate
```

# 5) Install TensorRT Python bindings (wheel)
Upgrade pip first:
```
pip install --upgrade pip
```

Install the TensorRT wheel:
```
python -m pip install -U pip
python -m pip install -U "$TRT_ROOT/python/tensorrt-10.15.1.29-cp312-none-linux_x86_64.whl"

export TRT_ROOT=$HOME/machgenai/TensorRT-10.15.1.29
export LD_LIBRARY_PATH=$TRT_ROOT/lib:${LD_LIBRARY_PATH:-}
```
(If you are running into issues, likely caused by Python version mismatch, run ls -1 "$TRT_ROOT/python/"*.whl to check what is available). You can also add these to bashrc so you don't have to re-export it every time.

Sanity check:
```
python -c "import tensorrt as trt; print('tensorrt:', trt.__version__)"
```

# 6) Install supporting Python deps
Pin pip to avoid strict version parsing issues (per your note), then install deps:
```
python -m pip install -U pip==23.3.2
python -m pip install -U torch
python -m pip install -U onnx onnxruntime onnxscript polygraphy
python -m pip install -U pycuda
```

# 7) Persist TensorRT env vars on venv activation
Create the activation hook dir:
```
mkdir -p .venv_trt/bin/activate.d
```

Create the hook script:
```
cat > .venv_trt/bin/activate.d/trt.sh <<'SH'
export TRT_ROOT=$HOME/machgenai/TensorRT-10.15.1.29
export PATH=$TRT_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$TRT_ROOT/lib:$LD_LIBRARY_PATH
SH
```
```
which trtexec || true
python -c "import tensorrt as trt; print('tensorrt:', trt.__version__)"
```

# 8) Validate if things are working
```
python test_trt.py
```

# 9) Setup diffusers, huggingface_hub etc.

```
# core
python -m pip install -U diffusers transformers accelerate safetensors

# Flux / image IO commonly needs these
python -m pip install -U huggingface_hub pillow numpy

python -m pip uninstall -y diffusers
python -m pip install -U "git+https://github.com/huggingface/diffusers.git"
```
