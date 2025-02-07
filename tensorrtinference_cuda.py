import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            # tensor_name = engine.get_tensor_name()
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            print(tensor_name)
            print(size)
            print(dtype)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropiate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        # torch.cuda.empty_cache()

        # Transfer input data to device
        # np.copyto(self.inputs[0].host, input_data.ravel())
        # cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        self.inputs[0].host = np.ascontiguousarray(input_data, dtype=np.float32)
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        # cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        # Synchronize the stream
        self.stream.synchronize()

        return self.outputs
