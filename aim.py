from mss import mss
import keyboard
import win32api
import win32con
import numpy as np
import datetime
from models.common import DetectMultiBackend
from utils.general import (check_img_size, cv2, non_max_suppression)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from pathlib import Path
import  torch

def screen_catch():
    with mss() as sct:
        monitor = {"top": 320,"left": 480,"width":1600,"height":960}
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        # cv2.imshow('img',img)
        # cv2.waitKey(100000)
        return img


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
device = select_device('0')
weights = ROOT / 'runs/train/exp8/weights/best.pt'
data = ROOT / 'archive/train/custom_data.yaml'
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz=(960,1600), s=stride)
bs = 1
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
teammate = [0,0]

def run(
        imgsz=(960,1600),
        conf_thres=0.45,
        iou_thres=0.45,
        max_det=10,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
):
    img0 = screen_catch()
    img = letterbox(img0, imgsz, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    #Inference
    t = datetime.datetime.now()
    pred = model(im, augment=augment, visualize=visualize)
    print(datetime.datetime.now() - t, 2)
    #NMS
    t = datetime.datetime.now()
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    print(datetime.datetime.now() - t,3)
    for i, det in enumerate(pred):
        if len(det):
            return det

def getxy(xyxy):
    x = int((xyxy[0] + xyxy[2])/2)
    y = int((xyxy[1] + xyxy[3])/2)
    return x,y


def is_teammate(c):
    return c == teammate[0] or c == teammate[1]


def i_am_ct():
    teammate[0] = 1
    teammate[1] = 2

def i_am_t():
    teammate[0] = 3
    teammate[1] = 4

def i_am_none():
    teammate[0] = 0
    teammate[1] = 0

def aim():
    print(teammate)
    det = run()
    best = (800,480,0,0)
    if det is not None:
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            if c < best[2] or conf.item() < best[3] or is_teammate(c) :
                continue
            x, y = getxy(xyxy)
            best = (x,y,c,conf.item())
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int((best[0]+480-2560/2)*0.4), int((best[1]+320-1600/2)*0.4), 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


if __name__ == '__main__':
    keyboard.add_hotkey('alt', aim)
    keyboard.add_hotkey('o',i_am_t)
    keyboard.add_hotkey('p',i_am_ct)
    keyboard.add_hotkey('o+p',i_am_none)
    keyboard.wait('q') #退出循环
