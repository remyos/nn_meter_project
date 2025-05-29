import os
import subprocess
import tensorflow as tf
import tf2onnx
import torch
import json
import onnx


from nn_meter.builder.backends import connect_backend
from nn_meter.builder.kernel_predictor_builder import generate_config_sample
from nn_meter.builder import builder_config, convert_models, profile_models  
from nn_meter.builder.kernel_predictor_builder.data_sampler.generator import KernelGenerator

import json

# -------- Helper: Extract input shape from ONNX --------
def get_input_shape_from_onnx(onnx_path):
    model = onnx.load(onnx_path)
    inputs = model.graph.input
    for inp in inputs:
        dims = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                dims.append(dim.dim_value)
            else:
                dims.append(1)  # fallback for dynamic dims
        return inp.name, dims
    return "input_image", [1, 3, 224, 224]  # fallback if none found

# -------- Generate DeepX JSON for each ONNX --------
def generate_deepx_jsons_for_onnx_dir(onnx_dir):
    """
    Generate DeepX-format JSON files per ONNX file using actual input shapes.
    """
    for filename in os.listdir(onnx_dir):
        if filename.endswith(".onnx"):
            onnx_path = os.path.join(onnx_dir, filename)
            base = os.path.splitext(filename)[0]
            json_path = os.path.join(onnx_dir, f"{base}.json")

            # Step 1: Read shape and input name
            input_name, input_shape = get_input_shape_from_onnx(onnx_path)
            C, H, W = input_shape[1], input_shape[2], input_shape[3]

            # Step 2: Build DeepX config
            config = {
                "inputs": {
                    input_name: input_shape
                },
            }

            # Step 3: Save
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[✓] Generated JSON for: {filename} → {input_shape}")
# -------- Step 1: Setup and Sample --------

workspace = "/home/remyos/nn-Meter/deepx_workspace"
builder_config.init(workspace)

backend = connect_backend(backend_name="deepx_npu_v2")

kernel_type = "conv-bn-relu"
sample_num = 10
mark = "test"

# Sample kernel configs
models = generate_config_sample(kernel_type, sample_num, mark=mark, sampling_mode="prior")


# -------- Step 3: Generate DeepX JSONs for all ONNX files --------
onnx_dir = os.path.join(workspace, "predictor_build", "kernels")
generate_deepx_jsons_for_onnx_dir(onnx_dir)

# convert the model to the needed format by backend, in order to increase efficiency when profiling on device.
models = convert_models(backend, f"{workspace}/predictor_build/results/{kernel_type}_{mark}.json")

# # run models with given backend and return latency of testcase models
profiled_results = profile_models(backend, models, mode='predbuild', have_converted=True,
                                   save_name=f"profiled_{kernel_type}.json")




