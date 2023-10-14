import tensorrt as trt
import os
import json
from calibrator import Calibrator
from datetime import datetime
from CalibDataLoader import CalibDataLoader
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_file_path", type=str, default="/media/ultralytics/model/yolov8n.onnx", help="ONNX path"
    )
    parser.add_argument(
        "--mode", type=str, default="INT8", help="precision selected"
    )
    parser.add_argument(
        "--calib_images_dir", type=str, default="/media/ultralytics/onnx2int8/calib_data", help="image dir of calibration"
    )
    parser.add_argument(
        "--calibration_table_path", type=str, default="/media/ultralytics/onnx2int8/weights/yolov8_calibration.cache", help="savd path of calibration_table"
    )
    parser.add_argument(
        "--engine_file_path", type=str, default="/media/ultralytics/onnx2int8/weights/yolov8.engine", help="saved engine path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="targer image width"
    )
    parser.add_argument(
        "--height", type=int, default=640, help="targer image height"
    )
    parser.add_argument(
        "--calib_count", type=int, default=4, help="the number of calib data"
    )
    return parser.parse_args()
    
def build_engine(onnx_file_path,mode,data_loader,calibration_table_path,engine_file_path):
    TRT_LOGGER=trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()#
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    assert os.path.exists(onnx_file_path),  "The onnx file {} is not found".format(onnx_file_path)
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("Building an engine from file {}, this may take a while...".format(onnx_file_path))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * (1 << 30))
    if mode == "INT8":
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = Calibrator(data_loader, calibration_table_path)
        config.int8_calibrator = calibrator
    elif mode == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_engine(network, config)
    if engine is None:
        print("Failed to create the engine")
        return None
    with open(engine_file_path, "wb") as f:  
        f.write(engine.serialize())

    return engine

def main(args):
    data_loader=CalibDataLoader(args.batch_size,args.width,args.height,  \
                                args.calib_count,args.calib_images_dir)
    build_engine(args.onnx_file_path,args.mode,    \
                 data_loader,args.calibration_table_path,args.engine_file_path)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

