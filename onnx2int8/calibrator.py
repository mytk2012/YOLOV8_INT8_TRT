import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
        self.cache_file = cache_file
        data_loader.reset()  

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        batch = self.data_loader.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.d_input, batch)

        return [self.d_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
