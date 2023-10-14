This project includes model conversion (yolov8 onnx to trt int8), and trt int8 inference. 

TRT inferencement can be implemented without pytorch.

Onnx generated:
yolov8 pt2onnx can refer to the conversion code provided by yolov8. 

Trt int8 generated:
CD onnx2int8 directory, execute main.py. 

Trt int8 inference:
CD int8_infer directory and execute infer_1batch.py.
