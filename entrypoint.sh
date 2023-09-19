#!/bin/bash

gfpgan_path="models/GFPGANv1.4.pth"
onnx_path="models/inswapper_128.onnx"

if [ ! -e "$gfpgan_path" ]; then
  wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth -P models
fi

if [ ! -e "$onnx_path" ]; then
  wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -P models
fi

python app.py