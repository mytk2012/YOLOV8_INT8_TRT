import glob
import os
import numpy as np
import cv2

class CalibDataLoader:
    def __init__(self, batch_size, width, height, calib_count, calib_images_dir):
        self.index = 0
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.calib_count = calib_count
        self.image_list = glob.glob(os.path.join(calib_images_dir, "*.png"))
        assert (
            len(self.image_list) > self.batch_size * self.calib_count
        ), "{} must contains more than {} images for calibration.".format(
            calib_images_dir, self.batch_size * self.calib_count
        )
        self.calibration_data = np.zeros((self.batch_size, 3, height, width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                image_path = self.image_list[i + self.index * self.batch_size]
                assert os.path.exists(image_path), "image {} not found!".format(image_path)
                image = cv2.imread(image_path)
                image = preprocess(image, self.width, self.height)
                self.calibration_data[i] = image
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.calib_count
    
def preprocess(raw_bgr_image,input_w,input_h): 
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
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate width and height and paddings  
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image

