from collections import namedtuple, OrderedDict

import torch
import numpy as np
import tensorrt as trt


class TensorRTInference:
    def __init__(self, engine_path, device='cuda:0', max_batch_size=32):
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.device = device
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, max_batch_size, device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

        print(self.bindings)
        print(self.input_names)
        print(self.output_names)

    def load_engine(self, engine_path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        return engine

    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)

        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)

        return names
        
    def get_bindings(self, engine, context, max_batch_size=32, device=None):
        '''build binddings
        '''
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True 
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings
    
    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs
    
    def __call__(self, blob):
        blob = {"images": torch.from_numpy(blob).to(self.device)}
        return self.run_torch(blob)
