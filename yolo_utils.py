import sys
import numpy as np 
import torch 

from torch.utils.data.dataloader import DataLoader

from tools.tv_reference.coco_eval import CocoEvaluator 
from tools.tv_reference.coco_utils import convert_to_coco_api 
# sys.path.append('..')
from models import post_process
from models import DarkNet
from cocodataset import Coco_dataset, yolo_collate_fn

from pprint import pprint

@torch.no_grad()
def evaluate(model, data_loader, device):
    '''
    use coco api to evalute results which model get out
    '''
    model.eval()
    model = model.to(device)

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='voc')
    coco_evaluator = CocoEvaluator(coco, iou_types=['bbox'], bbox_fmt='voc')

    for imgs, targets in data_loader:
        
        # imgs = imgs.div_(255.0).to(torch.float)
        imgs = imgs.to(device)
        for key in targets.keys():
            targets[key] = targets[key].to(device)
        predictions = model(imgs)
        # predictions becomes to list, each element is (predictions, 7)
        predictions = post_process(predictions)
        # for debug
        # for idx in range(len(predictions)):
        #     boxes = targets['boxes'][idx, ...]
        #     nonzero_ids = torch.nonzero(torch.prod(boxes, dim=1), as_tuple=True)[0]
        #     boxes = boxes[nonzero_ids]
        #     pprint(boxes)
        #     pprint(targets['labels'][idx][nonzero_ids])
        #     print('*'*75)
        #     pprint(predictions[idx])
        #     print('-'*75)
        #     print('-'*75)
        # for debug end
        # change targets to list
        targets = [{k: targets[k][i] for k in targets.keys()} for i in range(targets['area'].shape[0])]

        res = {}
        for target, prediction in zip(targets, predictions):
            
            if prediction.shape == torch.Size([]):
                continue
            image_id = target['image_id']
            boxes = prediction[:, :4]
            labels = prediction[:, -1]
            scores = prediction[:, 4]

            # boxes = boxes.unsqueeze(1)
            # labels = labels.unsqueeze(1)
            # scores = scores.unsqueeze(1)

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

    device = torch.device('cuda:1')
    model_val = DarkNet('./cfg/yolov4.cfg', False)
    model_val.load_weight('./data/yolov4.weights')

    dataset = Coco_dataset('/root/data/cocodataset', train=False)
    data_loader = DataLoader(dataset, 3, True, collate_fn=yolo_collate_fn)

    coco_evaluator = evaluate(model_val, data_loader, device)
    stats = coco_evaluator.coco_eval['bbox'].stats
    pprint(stats)
    # images, targets = next(iter(data_loader))
    # print(images.shape)
    # for imgs, targets in data_loader:
    #     pprint(imgs.shape)
