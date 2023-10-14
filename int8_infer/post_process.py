import numpy as np
import cv2
import random

class post():
    def __init__(self,width,height):
        self.input_w =width
        self.input_h =height
        self.conf_thres=0.25
        self.iou_thres=0.7
        self.classes=None
        self.nc=0
                
    def post_process(self,output,origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        boxes =self.non_max_suppression(output, origin_h, origin_w)
        if len(boxes):
            result_boxes,result_scores,result_classid=[],[],[]
            for i in range(len(boxes)):
                for j, pred in enumerate(boxes[i]):
                    result_boxes.append(pred[:4])
                    result_scores.append(pred[4])
                    result_classid.append(pred[5])
        else:
            result_boxes =np.array([])
            result_scores=np.array([])
            result_classid=np.array([])
        return result_boxes, result_scores, result_classid

    def non_max_suppression(self, prediction, origin_h, origin_w):
        bs = prediction.shape[0]
        nc=self.nc or (prediction.shape[1] - 4) 
        nm = prediction.shape[1] - nc - 4 
        mi = 4 + nc   #  nc是类别个数，nm是mask个数，mi是mask index
        xc = prediction[:, 4:mi].max(axis=1) > self.conf_thres 
        
        output=[np.zeros((0, 6 + nm))]*bs
        for xi, x in enumerate(prediction):  
            x = x.transpose(1,0)[xc[xi]]
            split_points = [4, nc, nm]
            split_arrays = []
            start = 0
            for split_point in split_points:
                split_arrays.append(x[:, start:start + split_point])
                start += split_point
            box, cls, mask = split_arrays[0],split_arrays[1],split_arrays[2]
            box=self.xywh2xyxy(origin_h,origin_w,box)
            conf=np.max(cls, axis=1, keepdims=True)
            j=np.argmax(cls, axis=1).astype(int)
            if j.shape[0]==0:
                continue
            else:
                j=j.reshape(j.shape[0],-1)
            x=np.concatenate((box,conf,j,mask),axis=1)[conf.flatten()> self.conf_thres]
            n=x.shape[0]
            if not n:  # no boxes
                continue
            x=x[np.argsort(x[:, 4], axis=0)[::-1][:30000]]
            boxes, scores = x[:, :4], x[:, 4]
            i=self.nms(boxes, scores, self.iou_thres)
            output[xi]=x[i] 
        return output

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def bbox_iou(self,box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - intersection
        iou = intersection / union
        return iou

    def nms(self,boxes, scores, iou_threshold):
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]

        keep = []

        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]

        keep = []

        while len(boxes) > 0:
            keep.append(order[0])
            iou = self.bbox_iou(boxes[0], boxes[1:])
            mask = iou <= iou_threshold
            boxes = boxes[1:][mask]
            scores = scores[1:][mask]
            order = order[1:][mask]

        return keep
    
    def plot_box(self, box,img,save_img_tag,save_img_name, save_path=None,label=None):
        """
        description: Plots one bounding box on image img,
                    this function comes from YoLov5 project.
        param: 
            x:      a box likes [x1,y1,x2,y2]
            img:    a opencv image object
            color:  color to draw rectangle, such as (0,255,0)
            label:  str

        return:
            no return
        """
        line_thickness=None
        color=None
        tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

        tf = max(tl - 1, 1) 
        cv2.putText(
            img,
            label,
            (c1[0], max(c1[1] - 10,10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            tl / 3,
            (0, 0, 255),
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
        if save_img_tag:
            new_img_name=save_path+save_img_name.split('/')[-1]
            print(new_img_name)
            cv2.imwrite(new_img_name,img)
        
