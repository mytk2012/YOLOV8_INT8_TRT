This project includes model conversion (yolov8 onnx to trt int8), and trt int8 inference. 

TRT inferencement can be implemented without pytorch. And its inference can be up to 2000fps!

Install
If you wanna convert pt2onnx,pleace refer to yolov8 https://github.com/ultralytics/ultralytics.
When you wanna start convert onnx2int8trt or int8trt inference, you can refer to the import library to install it.

Model Convert
Onnx generated:
yolov8 pt2onnx can refer to the conversion code provided by yolov8. 
Trt int8 generated:
CD onnx2int8 directory, execute main.py. 

Model Inference
Trt int8 inference:
CD int8_infer directory and execute infer_1batch.py.

