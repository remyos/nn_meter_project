# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
from nn_meter.builder.backends import BaseBackend, BaseParser, BaseProfiler

import subprocess
import tensorflow as tf
import tf2onnx
import torch

class MyParser(BaseParser):...
class MyProfiler(BaseProfiler):... 
    
class DeepxNPUBackend(BaseBackend):
    parser_class = MyParser
    profiler_class = MyProfiler
    
    def __init__(self, configs):
        """ class initialization with required configs
        """
        self.configs = configs
        self.update_configs()
        if self.parser_class:
            self.parser = self.parser_class(**self.parser_kwargs)
        if self.profiler_class:
            self.profiler = self.profiler_class(**self.profiler_kwargs)

    def update_configs(self):
        """ update the config parameters for the backend
        """
        self.parser_kwargs = {}
        self.profiler_kwargs = {}
    
    def convert_model(self, model_path, save_path, input_shape=None):
        """
        Compile an existing ONNX model using DeepX dx_com.

        Parameters:
        - model_path (str): Path to the input ONNX model (*.onnx)
        - save_path (str): Directory to save the compiled .dxnn model
        - input_shape (ignored): Not used here, required only for model generation

        Returns:
        - str: Path to the compiled .dxnn file
        """

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        base_name = os.path.splitext(os.path.basename(model_path))[0]
        config_path = os.path.join(os.path.dirname(model_path), f"{base_name}.json")
        output_dir = os.path.join(save_path, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Check ONNX and config existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Expected config JSON not found: {config_path}")

        # Step 2: Compile with dx_com
        cmd = [
            "/home/remyos/dx_com_M1A_v1.24.0/dx_com/dx_com",
            "-m", model_path,
            "-c", config_path,
            "-o", output_dir,
            "--shrink"
        ]

        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("DeepX compile output:", result.stdout.decode())
        except subprocess.CalledProcessError as e:
            print(f"[{base_name}] DeepX compile failed:")
            print(e.stderr.decode())  # <-- very important to see why it failed
            raise RuntimeError(f"DeepX model compilation failed for {base_name}")


        # Step 3: Return compiled path
        dxnn_path = os.path.join(output_dir, f"{base_name}.dxnn")
        if not os.path.exists(dxnn_path):
            raise FileNotFoundError(f"Compiled .dxnn model not found at: {dxnn_path}")

        return dxnn_path

    def profile_model_file(self, model_path, save_path, input_shape = None, metrics = ['latency'], **kwargs):
        """ load model by model file path, convert model file, and run ``self.profile()``
        @params:

        model_path: the path of model waiting to profile
        
        save_path: folder to save the converted model
        
        input_shape: the shape of input tensor for inference, a random tensor according to the shape will be 
            generated and used
        """
        converted_model = self.convert_model(model_path, save_path, input_shape)
        res = self.profile(converted_model, metrics, input_shape=input_shape, **kwargs)
        return res

    
    def test_connection(self):
        # Optional: check if the device is connected and working
        print("hello backend !")
        logging.info("hello backend !")
