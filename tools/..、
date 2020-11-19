import sys
import numpy as np 
import torch 

from torch.utils.data.dataloader import DataLoader

from tv_reference.coco_eval import CocoEvaluator 
from tv_reference.coco_utils import convert_to_coco_api 
sys.path.append('..')
from models import post_process
from models import DarkNet
from cocodataset import Coco_dataset

from pprint import pprint

@torch.no_grad()
def evaluate(model, data_loader, device):
    '''
    use coco api to evalute results which model get out
    '''
    model.eval()

    # coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='voc')
    # coco_evaluator = CocoEvaluator(coco, iou_types=['bbox'], bbox_fmt='coco')

    for imgs, targets in data_loader:
        
        imgs = imgs.div_(255.0)
        predictions = model(imgs)
        # predictions becomes to list, each element is (predictions, 7)
        predictions = post_process(imgs)
        # change targets to list
        targets = [{k: targets[k][i] for k in targets.keys()} for i in range(targets.shape[0])]

        res = {}
        for target, prediction in zip(targets, predictions):
            
            image_id = target['image_id']
            boxes = prediction[:, :4]
            labels = prediction[:, -1]
            scores = prediction[:, 4]

            res[image_id.item()] = {
                'boxes': boxes,
                'labels': labels,
                'scores': scores
            }
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator

if __name__ == "__main__":

    model_val = DarkNet('../cfg/yolov4.cfg', False)
    model_val.load_weight('../data/yolov4.weights')

    dataset = Coco_dataset('/home/baodi/data/cocodataset', train=False)
    data_loader = DataLoader(dataset, 4, True, num_workers=4)

    coco_evaluator = evaluate(model_val, data_loader, torch.device('cuda:1'))
    # images, targets = next(iter(data_loader))
    # print(images.shape)
    # for imgs, targets in data_loader:
    #     pprint(imgs.shape)
