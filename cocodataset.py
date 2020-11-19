import os

import torch 
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms 

from pycocotools.coco import COCO

import numpy as np 
import cv2
import random

def get_class_name(cat):
    class_names = load_class_names("/home/baodi/yolo_projects/project_yolov4/data/coco.names")
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat, class_names[cat]

def load_class_names(namesfile):
    '''
    read coco.name for classes
    '''
    class_names= []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def save_image_boxes(img, boxes, name='imageboxes.jpg', obj_names=None):
    '''
    img is numpy array
    boxes shape is like (n, 4) and is tensor
    '''
    boxes = boxes.numpy()
    show_img = img.copy()
    for i in range(boxes.shape[0]):
        rect = boxes[i].astype(np.int)
        start = (rect[0], rect[1])
        end = (rect[2], rect[3])
        def rand_color():
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = rand_color()
        cv2.rectangle(show_img, start, end, color, 1)
        if obj_names is not None:
            org = (rect[0], rect[1]+10)
            font = 1
            fontScale = 1
            show_img = cv2.putText(show_img, obj_names[i], org, font, fontScale, color, 1, cv2.LINE_AA) 
    cv2.imwrite(name, show_img)

def get_original_img_boxes(new_img, origin_size, target):
    '''
    new_img size : (H, W, C)
    origin_size tuple of int (ori_H, ori_W)
    target is dict
    '''
    new_size = new_img.shape[0]
    ori_h, ori_w = origin_size
    
    # 1. get rid of space
    ratio = min(new_size / ori_h, new_size / ori_w)
    new_h = int(ori_h * ratio)
    new_w = int(ori_w * ratio)
    space = np.array([new_size - new_h, new_size - new_w]) // 2
    canvas = np.zeros((new_h, new_w, new_img.shape[2]))
    canvas = np.array(new_img[space[0] : space[0] + new_h, space[1] : space[1] + new_w, :], copy=True)
    # 2. resize canvas
    canvas = cv2.resize(canvas, (ori_w, ori_h), interpolation=cv2.INTER_CUBIC)
    # 3. get right target
    boxes = target['boxes']
    # boxes /= ratio
    boxes[:, [0, 2]] -= space[1]
    boxes[:, [1, 3]] -= space[0]
    boxes /= ratio
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    target['boxes'] = boxes
    target['area'] = areas

    return canvas, target

def resize_transform(size, img, target):
    '''
    size is int, is the new width and new height
    '''
    # 1. resize the image
    old_h, old_w, old_c = img.shape
    
    ratio = min(size / old_h, size / old_w)
    new_w = int(old_w * ratio)
    new_h = int(old_h * ratio)
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    space = np.array([size - new_h, size - new_w]) // 2
    canvas = np.zeros((size, size, old_c))
    canvas[:, :, :] = np.mean(img)
    canvas[space[0] : space[0] + new_h, space[1] : space[1] + new_w, :] = new_img

    # 2. get the boxes
    if 'boxes' in target:
        boxes = target['boxes']
        boxes *= ratio
        boxes[:, [0, 2]] += space[1]
        boxes[:, [1, 3]] += space[0]
        target['boxes'] = boxes

        # 3. get areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target['area'] = areas
    return canvas, target

def crop_transform(img, target, crop_size=None):
    '''
    crop size is like (lefttop_x, lefttop_y, w, h)
    image become size like crop_size
    '''
    if crop_size is None:
        return img, target 
    
    # get crop img
    canvas = np.zeros((crop_size[3], crop_size[2], img.shape[2]))
    canvas[:, :, :] = np.mean(img)
    def rect_intersection(rect1, rect2):
        '''
        rect1 and rect2 as (x1, y1, x2, y2)
        '''
        lt_x = max(rect1[0], rect2[0])
        lt_y = max(rect1[1], rect2[1])
        rb_x = min(rect1[2], rect2[2])
        rb_y = min(rect1[3], rect2[3])
        return np.array([lt_x, lt_y, rb_x, rb_y], dtype=np.int)

    img_rect = np.array([0, 0, img.shape[1], img.shape[0]], dtype=np.float)
    crop_rect = np.array([crop_size[0], crop_size[1], crop_size[0]+crop_size[2], crop_size[1]+crop_size[3]], dtype=np.float)
    iter_rect = rect_intersection(img_rect, crop_rect)

    dst_rect = np.array([max(0, -crop_size[0]), max(0, -crop_size[1]), 
                         max(0, -crop_size[0])+iter_rect[2]-iter_rect[0], max(0, -crop_size[1])+iter_rect[3]-iter_rect[1]], dtype=np.int)

    canvas[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2], :] = img[iter_rect[1]:iter_rect[3], iter_rect[0]:iter_rect[2], :]

    # get right target
    if 'boxes' in target:
        boxes = target['boxes']
        boxes[:, [0, 2]] -= crop_size[0]
        boxes[:, [1, 3]] -= crop_size[1]
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], dst_rect[0], dst_rect[2])
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], dst_rect[1], dst_rect[3])

        # get rid of boxes out of range
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        valid_idx = torch.nonzero(areas, as_tuple=True)[0].numpy()

        target['boxes'] = boxes[valid_idx, :]
        target['labels'] = target['labels'][valid_idx]
        target['area'] = areas[valid_idx]
        target['iscrowd'] = target['iscrowd'][valid_idx]
        # target['names'] = [target['names'][i] for i in valid_idx]

    return canvas, target


class Coco_dataset(Dataset):
    def __init__(self, coco_root, train=True, resize=608, max_objnum=100):
        super(Coco_dataset, self).__init__()
        self.root = coco_root
        self.size = resize
        self.max_obj = max_objnum
        if train:
            self.data_mode = 'train'
            self.img_path = os.path.join(self.root, 'train2017')
        else:
            self.data_mode = 'val'
            self.img_path = os.path.join(self.root, 'val2017')
        
        self.img_ids = []
        self.targets = []
        self._load_coco_data()

    def __getitem__(self, idx):
        img, target = self._get_val_item(idx)
        if self.data_mode == 'val':
            img, target = resize_transform(self.size, img, target)
            # return img, target
        else:
            crop_w = random.randint(img.shape[1] // 2, img.shape[1])
            crop_h = random.randint(img.shape[0] // 2, img.shape[0])
            crop_x = random.randint(0, img.shape[1] // 2)
            crop_y = random.randint(0, img.shape[0] // 2)
            crop_size = torch.tensor([crop_x, crop_y, crop_w, crop_h], dtype=torch.int)
            img, target = crop_transform(img, target, crop_size)
            img, target = resize_transform(self.size, img, target)
        # get align at obj nums and img to tensor
        img = transforms.ToTensor()(img)
        img = img.to(torch.float)
        return img, target

    def _load_coco_data(self):
        if self.data_mode == 'train':
            ann_file = os.path.join(self.root, 'annotations/instances_train2017.json')
        else:
            ann_file = os.path.join(self.root, 'annotations/instances_val2017.json')
        self.coco = COCO(ann_file)
        img_ids = self.coco.getImgIds()
        for i, img_id in enumerate(img_ids):
            self.img_ids.append(img_id)
        
    def _get_val_item(self, idx):
        img_id = self.img_ids[idx]
        img_file = self.coco.loadImgs(img_id)[0]['file_name']
        img_file = os.path.join(self.img_path, img_file)
        img = cv2.imread(img_file)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bboxes = []
        labels = []
        areas = []
        target = {}
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        for i, ann_id in enumerate(ann_ids):
            ann = self.coco.loadAnns(ann_id)
            bbox = ann[0]['bbox']
            bbox = torch.tensor(bbox, dtype=torch.float32)
            bbox[2:] += bbox[:2]
            bbox = bbox.unsqueeze(0)
            bboxes.append(bbox)

            # get labels
            cat_id = ann[0]['category_id']
            # get labels' name 
            cat_id, cat_name = get_class_name(cat_id)
            labels.append(torch.tensor([cat_id], dtype=torch.int64))
            area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
            areas.append(area)
        if len(bboxes) > 0:
            target['boxes'] = torch.cat(bboxes, dim=0)
            target['labels'] = torch.cat(labels, dim=0)
            target['area'] = torch.cat(areas, dim=0)

            target['iscrowd'] = torch.zeros((len(bboxes),))
        else :
            target['boxes'] = torch.zeros((1, 4), dtype=torch.float)
            target['labels'] = torch.zeros((1, ), dtype=torch.float)
            target['area'] = torch.zeros((1, ), dtype=torch.float)

            target['iscrowd'] = torch.zeros((1, ), dtype=torch.float)
        target['image_id'] = torch.tensor([img_id])
        # img, target = resize_transform(self.size, img, target)
        return img, target

    def __len__(self):
        return len(self.img_ids)

def yolo_collate_fn(batch):
    imgs, targets = batch[:][0], batch[:][1]
    imgs = [batch[i][0] for i in range(len(batch))]
    targets = [batch[i][1] for i in range(len(batch))]
    imgs = torch.stack(imgs, dim=0).div_(255.0)
    max_obj = 100
    labels = {}
    keys_to_expand = ['boxes', 'labels', 'area', 'iscrowd']
    if isinstance(targets[0], dict):
        for target in targets:
            for key in keys_to_expand:
                if key == 'boxes':
                    obj_num = target['boxes'].shape[0]
                    boxes = torch.zeros((max_obj, 4), dtype=torch.float)
                    boxes[:min(obj_num, max_obj), :] = target['boxes'][:min(obj_num, max_obj), :]
                    target['boxes'] = boxes
                else:
                    obj_num = target[key].shape[0]
                    key_values = torch.zeros((max_obj, ), dtype=torch.float)
                    key_values[:min(obj_num, max_obj)] = target[key][:min(obj_num, max_obj)]
                    target[key] = key_values

        for key in targets[0].keys():
            labels[key] = torch.stack([targets[i][key] for i in range(len(targets))], dim=0)
    return imgs, labels


if __name__ == "__main__":
    dataset = Coco_dataset('/home/baodi/data/cocodataset', train=False)
    num = random.randint(0, len(dataset))
    max_obj = 0
    max_id = 0
    # for i in range(len(dataset)):
    #     # print(i)
    #     _, target = dataset[i]
    #     if 'boxes' in target:
    #         obj_num = target['boxes'].shape[0]
    #         if max_obj < obj_num:
    #             max_obj = obj_num
    #             max_id = i

    # print("max obj is {}, id: {}".format(max_obj, max_id))
    img, target = dataset[5]
    # class_names = load_class_names("./data/coco.names")
    # target_names = [class_names[i] for i in target['labels']]
    # new_img, new_target = crop_transform(img, target, crop_size=[-10, 100, 650, 250])
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    save_image_boxes(img, target['boxes'], 'imgwithname.jpg')
    for key in target.keys():
        print(key,target[key].shape)
    # new_img, new_target = resize_transform(608, img, target)
    
    # save_image_boxes(new_img, new_target['boxes'], 'newimgwithname.jpg', new_target['names'])

    # resize_img, resize_target = get_original_img_boxes(new_img, (img.shape[0], img.shape[1]), new_target)
    # cv2.imwrite('resize.jpg', resize_img)
    # save_image_boxes(resize_img, resize_target['boxes'], "resizeimgwithname.jpg", resize_target['names'])


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=yolo_collate_fn)
    imgs, targets = next(iter(dataloader))
    print(imgs.shape)
    print([key for key in targets.keys()])
    print([targets[key].shape for key in targets.keys()])
    # print(targets['image_id'])