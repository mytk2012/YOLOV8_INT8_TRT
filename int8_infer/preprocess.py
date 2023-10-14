import cv2
import numpy as np

def preprocess(raw_bgr_image,input_w,input_h): #tensorrtx
    """
    description: Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    image_raw = raw_bgr_image
    h, w, c = image_raw.shape#(480, 640, 3)
    if (np.ones((h,w,c))==image_raw).all():#如果是预热的话
        image=image_raw
    else:#正式推理的时候
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate width and height and paddings   input_w是目标width
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:#(1.0, 1.3333333333333333)
        tw = input_w#640
        th = int(r_w * h)#480
        tx1 = tx2 = 0#0
        ty1 = int((input_h - th) / 2)#80
        ty2 = input_h - th - ty1#80
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))#(480,640,3)
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
    )#(640, 640, 3)  pre_transform执行完成
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])#(3, 640, 640)
    # CHW to NCHW format
    # image = np.expand_dims(image, axis=0)#(1, 3, 640, 640)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)#(1, 3, 640, 640)
    return image, image_raw, h, w #((1, 3, 640, 640), (480, 640, 3), 480, 640)