import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import pycuda.gpuarray as gpuarray

from preprocess import preprocess
from post_process import post
import os
import argparse

class YOLOV8_trt(object):
    def __init__(self,image,engine_path,width,height):
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.width=width
        self.height=height
        self.categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]
        self.image=image

    def infer(self,save_img_name,save_img_flag,save_path):
        '''前处理'''
        input_image,image_raw, origin_h, origin_w= preprocess(self.image,self.width,self.height)
        '''推理'''
        image_width = input_image.shape[1]
        image_height = input_image.shape[2]
        with self.engine.create_execution_context() as context:
            context.set_binding_shape(self.engine.get_binding_index("images"), (1, 3, image_height, image_width))
            bindings = []
            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                if self.engine.binding_is_input(binding):
                    '''输入binding'''
                    input_buffer = np.ascontiguousarray(input_image)
                    input_memory = cuda.mem_alloc(input_image.nbytes)
                    bindings.append(int(input_memory))
                else:
                    '''输出binding'''
                    output_shape=context.get_binding_shape(binding_idx)
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))
                    
            stream = cuda.Stream()
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)   
            start_time=time.time()
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            end_time=time.time()
            time_consume=end_time-start_time
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream) 
            stream.synchronize()
            output_buffer=output_buffer.reshape(output_shape)#array
            '''后处理'''
            post_process=post(self.width,self.height)
            result_boxes, result_scores, result_classid = post_process.post_process(
                    output_buffer, origin_h, origin_w
                )
            if len(result_boxes)>0:
                for j in range(len(result_boxes)):
                    box = result_boxes[j]
                    post_process.plot_box(
                        box,
                        image_raw,
                        save_img_flag,
                        save_img_name,
                        save_path,
                        label="{}:{:.2f}".format(
                            self.categories[int(result_classid[j])], result_scores[j]
                        ),
                    )
                print("Detected")
            else:
                if save_img_flag:
                    print("Nothing Detected")
        return time_consume 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine_path", type=str, default="/media/ultralytics/onnx2int8/weights/yolov8.engine", help="engine path"
    )
    parser.add_argument(
        "--image_dir", type=str, default="/media/ultralytics/data/", help="inference image dir"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="image width"
    )
    parser.add_argument(
        "--height", type=int, default=640, help="image height"
    )
    parser.add_argument(
        "--warm_image_path", type=str, default="/media/ultralytics/data/test1.png", help="warm up init"
    )
    parser.add_argument(
        "--save_path", type=str, default="/media/ultralytics/demo/", help="warm up init"
    )
    return parser.parse_args()

def main(args):
    image_num=len(os.listdir(args.image_dir))
    init_img=cv2.imread(args.warm_image_path)
    
    #warm_up
    save_img_tag=False
    init_h, init_w, init_c = init_img.shape
    warm_img=np.ones((init_h,init_w,init_c))
    warmup_model=YOLOV8_trt(warm_img,args.engine_path,args.width,args.height) 
    warmup_time=warmup_model.infer("",save_img_tag,args.save_path)
    
    #inference
    save_img_tag=True
    time_=0
    for file in os.listdir(args.image_dir):
        image_path=args.image_dir+file
        img=cv2.imread(image_path)
        Detect_model=YOLOV8_trt(img,args.engine_path,args.width,args.height)
        total_time=Detect_model.infer(image_path,save_img_tag,args.save_path)
        time_+=total_time
    print("time cost:",time_*1000/(image_num),"ms")
    
if __name__=="__main__":
    args = parse_args()
    main(args)
    
    
    
    
    
    
    
    
        

        