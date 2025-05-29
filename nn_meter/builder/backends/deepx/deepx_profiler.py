# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os

from ..interface import BaseBackend, BaseParser, BaseProfiler

class BaseProfiler:
    """
    Specify the profiling command of the backend. A profiler contains commands to push the model to mobile device, run the model 
    on the mobile device, get stdout from the mobile device, and related operations. 
    """
    def profile(self):
        """ Main steps of ``Profiler.profile()`` includes 1) push the model file to edge devices, 2) run models in required times
        and get back running results. Return the running results on edge device.
        """
        output = ''
        return output
