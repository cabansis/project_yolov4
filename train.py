from cocodataset import Coco_dataset, get_original_img_boxes, load_class_names, save_image_boxes, yolo_collate_fn
from models import DarkNet, get_iou, post_process
from yolo_utils import evaluate

import os 
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.tv_reference.coco_eval import CocoEvaluator
from tools.tv_reference.coco_utils import convert_to_coco_api

from pprint import pprint

class_names = load_class_names('./data/coco.names')
class Yolo_loss(nn.Module):
    '''
    get yolo loss for outfeatures
    anchors is all the 9 anchors
    anchor_masks is the mask of 3 output featuremaps
    '''
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch_size=1, iou_threshold=0.5):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.img_size = 608
        self.strides = [8, 16, 32]
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.iou_threshold = iou_threshold

        self.ref_anchors, self.grid_x, self.grid_y, self.all_anchors, self.masked_anchors = [], [], [], [], []
        self.anchors_w, self.anchors_h = [], []

        for i in range(n_anchors):
            fsize = int(self.img_size // self.strides[i])
            # 1. get all anchors at each stride
            one_all_anchors = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            one_all_anchors = np.array(one_all_anchors, dtype=np.float32)
            # one_all_anchors = torch.from_numpy(one_all_anchors)
            self.all_anchors.append(one_all_anchors)
            # 2. get masked anchors at each stride
            one_masked_anchors = np.array([one_all_anchors[j] for j in self.anchor_masks[i]], dtype=np.float32)
            # one_masked_anchors = torch.from_numpy(one_masked_anchors)
            self.masked_anchors.append(one_masked_anchors)
            # 3. get ref anchors at each stride
            ref_zeros = np.zeros_like(one_all_anchors, dtype=np.float32)
            one_ref_anchors = np.concatenate((ref_zeros, one_all_anchors), axis=1)
            one_ref_anchors = torch.from_numpy(one_ref_anchors).to(torch.float).to(self.device)
            self.ref_anchors.append(one_ref_anchors)
            # 4. get grid_x and grid_y at each stride
            one_grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch_size, 3, fsize, 1).to(self.device)
            one_grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch_size, 3, fsize, 1).permute(0, 1, 3, 2).to(self.device)
            self.grid_x.append(one_grid_x)
            self.grid_y.append(one_grid_y)
            # 5. get anchors_w and anchors_h at each stride
            one_anchors_w = torch.from_numpy(one_masked_anchors[:, 0]).repeat(batch_size, fsize, fsize, 1).permute(0, 3, 1, 2).contiguous().to(torch.float)
            one_anchors_h = torch.from_numpy(one_masked_anchors[:, 1]).repeat(batch_size, fsize, fsize, 1).permute(0, 3, 1, 2).contiguous().to(torch.float)
            one_anchors_w = one_anchors_w.to(self.device)
            one_anchors_h = one_anchors_h.to(self.device)
            self.anchors_w.append(one_anchors_w)
            self.anchors_h.append(one_anchors_h)
        # pprint(one_all_anchors)
        # pprint(one_masked_anchors)
        # pprint(one_ref_anchors)
        # pprint(one_anchors_w[0][0])

    def build_target(self, pred, labels, batch_size, fsize, n_ch, output_id):
        '''
        pred shape: (batch, n_anchors, fsize, fsize, 5+n_classes)
        labels shape: (batch, 100, 5) --> 100 is the output size of dataloader, 5 is (x1, y1, x2, y2, class_id)
        output:
            all the outputs are from the unique feature map
            obj_mask: (batch, n_anchors, fsize, fsize, 1) --> judge if obj in that anchor, get loss of obj prediction
            tgt_mask: (batch, n_anchors, fsize, fsize, 4+n_classes) --> get loss of boxes and classes
            tgt_scale: (batch, n_anchors, fsize, fsize, 2) --> multiply to w and h
            target: (batch, n_anchors, fsize, fsize, 5+n_classes) --> truth feature map from ground truth 
        '''
        obj_mask = torch.zeros((batch_size, self.n_anchors, fsize, fsize), dtype=torch.float).to(self.device)
        tgt_mask = torch.zeros((batch_size, self.n_anchors, fsize, fsize, 4+self.n_classes), dtype=torch.float).to(self.device)
        tgt_scale = torch.zeros((batch_size, self.n_anchors, fsize, fsize, 2), dtype=torch.float).to(self.device)
        target = torch.zeros((batch_size, self.n_anchors, fsize, fsize, 5+self.n_classes), dtype=torch.float).to(self.device)

        nlabels = (labels.sum(dim=2) > 0).sum(dim=1)

        truth_x_all = (labels[..., 0] + labels[..., 2]) / 2 / self.strides[output_id]
        truth_y_all = (labels[..., 1] + labels[..., 3]) / 2 / self.strides[output_id]
        truth_w_all = (labels[..., 2] - labels[..., 0]) / self.strides[output_id]
        truth_h_all = (labels[..., 3] - labels[..., 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.floor()
        truth_j_all = truth_y_all.floor()

        for b in range(batch_size):
            n_objs = nlabels[b]
            if n_objs == 0:
                continue
            pred_xyxy = pred[b].view(-1, 4)
            pred_xyxy[:, :2] -= pred_xyxy[:, 2:] / 2
            pred_xyxy[:, 2:] += pred_xyxy[:, :2]

            truth_boxes = torch.zeros((n_objs, 4), dtype=torch.float, device=self.device)
            truth_boxes[:, 2] = truth_w_all[b, :n_objs]
            truth_boxes[:, 3] = truth_h_all[b, :n_objs]
            # get the most suitable anchor for each truth box
            truth_anchors_ious = get_iou(truth_boxes, self.ref_anchors[output_id])
            match_all_anchors = torch.argmax(truth_anchors_ious, dim=1)
            match_anchors = match_all_anchors % 3
            match_anchors_mask = ((match_all_anchors[:] == self.anchor_masks[output_id][0]) | 
                                  (match_all_anchors[:] == self.anchor_masks[output_id][1]) |
                                  (match_all_anchors[:] == self.anchor_masks[output_id][2]))
            # pprint(sum(match_anchors_mask))
            if sum(match_anchors_mask) < 1:
                continue
            truth_boxes[:, 0] = truth_x_all[b, :n_objs]
            truth_boxes[:, 1] = truth_y_all[b, :n_objs]
            truth_boxes[:, :2] -= truth_boxes[:, 2:] / 2
            truth_boxes[:, 2:] += truth_boxes[:, :2]
            pred_truth_ious = get_iou(pred_xyxy, truth_boxes)

            pred_truth_ious_best, _ = torch.max(pred_truth_ious, dim=1)
            pred_truth_ious_valid = pred_truth_ious_best >= self.iou_threshold
            pred_truth_ious_valid = pred_truth_ious_valid.view(pred[b].shape[:-1])
            # pprint(pred_truth_ious_best.shape)
            obj_mask[b] = ~pred_truth_ious_valid

            for obj_i, anchor_i in enumerate(match_anchors):
                if match_anchors_mask[obj_i] == 0:
                    continue
                i = truth_i_all[b, obj_i].to(torch.int16).cpu().numpy()
                j = truth_j_all[b, obj_i].to(torch.int16).cpu().numpy()
                anchor_i = anchor_i.to(torch.int16).cpu().numpy()

                obj_mask[b, anchor_i, j, i] = 1
                tgt_mask[b, anchor_i, j, i, :] = 1
                target[b, anchor_i, j, i, 0] = truth_x_all[b, obj_i]
                target[b, anchor_i, j, i, 1] = truth_y_all[b, obj_i]
                target[b, anchor_i, j, i, 2] = truth_w_all[b, obj_i]
                target[b, anchor_i, j, i, 3] = truth_h_all[b, obj_i]
                target[b, anchor_i, j, i, 4] = 1
                target[b, anchor_i, j, i, 5+labels[b, obj_i, 4].to(torch.int16).cpu().numpy()] = 1
                tgt_scale[b, anchor_i, j, i, :] = torch.sqrt(2 - truth_w_all[b, obj_i] * truth_h_all[b, obj_i] / fsize / fsize)

        return obj_mask, tgt_mask, tgt_scale, target

    def iou_loss(self, pred_boxes, gt_boxes, iou_type='ciou', fsize=13):
        '''
        get iou loss 
        pred_boxes and gt_boxes are both (cx, cy, w, h)
        '''
        assert(pred_boxes.shape[0] == gt_boxes.shape[0])
        target_scale = torch.zeros((gt_boxes.shape[0], 1), dtype=torch.float32, device=pred_boxes.device)
        target_scale = 2 - (gt_boxes[..., 2] * gt_boxes[..., 3]) / fsize**2

        pred_tl = pred_boxes[..., :2] - pred_boxes[..., 2:4] / 2
        pred_br = pred_boxes[..., :2] + pred_boxes[..., 2:4] / 2
        gt_tl   = gt_boxes[..., :2] - gt_boxes[..., 2:4] / 2
        gt_br   = gt_boxes[..., :2] + gt_boxes[..., 2:4] / 2

        inter_tl = torch.max(pred_tl, gt_tl)
        inter_br = torch.max(pred_br, gt_br)

        conv_tl = torch.min(pred_tl, gt_tl)
        conv_br = torch.min(pred_br, gt_br)

        inter_areas = torch.prod((inter_br - inter_tl), dim=1)
        conv_areas = torch.prod((conv_br - conv_tl), dim=1)

        pred_areas = pred_boxes[..., 2] * pred_boxes[..., 3]
        gt_areas = gt_boxes[..., 2] * gt_boxes[..., 3]

        uion_areas = pred_areas + gt_areas - inter_areas + 1e-16

        ious = inter_areas / uion_areas

        if iou_type == 'giou':
            gious = ious - uion_areas / (conv_areas + 1e-16)
            return torch.sum((1 - gious)*target_scale)
        elif iou_type == 'diou' or iou_type == 'ciou':
            conv_distance = torch.pow(conv_br - conv_tl, 2).sum(dim=-1) + 1e-16
            center_distance = torch.pow(pred_boxes[..., :2] - gt_boxes[..., :2], 2).sum(dim=-1)
            if iou_type == 'diou':
                dious = ious - center_distance / conv_distance
                return torch.sum((1-dious)*target_scale)
            elif iou_type == 'ciou':
                v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(pred_boxes[..., 2] / pred_boxes[..., 3]) - torch.atan(gt_boxes[..., 2] / gt_boxes[..., 3]), 2)
                with torch.no_grad():
                    alpha = v / (1 - ious + v + 1e-16)
                cious = ious - center_distance / conv_distance - alpha * v
                cious = torch.clamp(cious, min=-1.0, max=1.0)
                return torch.sum((1-cious)*target_scale)



    def forward(self, outfeatures, truth_target):
        assert(len(outfeatures) == 3)
        loss, loss_obj, loss_class, loss_iou = 0, 0, 0, 0
        boxes = truth_target['boxes'].to(device)
        classes = truth_target['labels'].unsqueeze(2).to(device)
        labels = torch.cat((boxes, classes), dim=2)
        # print(labels[0, :, :])
        for output_id, output in enumerate(outfeatures):
            
            batchsize = output.shape[0]
            fsize = output.shape[-1]
            n_ch = 5 + self.n_classes
            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            # output shape: (batchsize, n_anchors, fsize, fsize, 5+n_classes)
            output = output.permute(0, 1, 3, 4, 2)

            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])
            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchors_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchors_h[output_id]
            
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)
            
            output[..., 4] *= obj_mask
            output[..., np.r_[:4, 5:n_ch]] *= tgt_mask
            
            target[..., 4] *= obj_mask
            target[..., np.r_[:4, 5:n_ch]] *= tgt_mask
            
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_class += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            # input_label = output[..., 5:].reshape(-1, self.n_classes)
            # class_label = torch.argmax(target[..., 5:], dim=-1).view(input_label.shape[0],)
            # loss_class += F.cross_entropy(input=input_label, target=class_label, reduction='sum')
            
            # iou losses for boxes
            target_boxes = target[..., :4].clone()
            target_boxes = target_boxes.contiguous().view(-1, 4)
            
            output_boxes = pred.contiguous().view(-1, 4)

            target_boxes_nonzero = torch.nonzero(torch.sum(target_boxes, dim=-1), as_tuple=True)[0]
            target_boxes = target_boxes[target_boxes_nonzero]
            # output_boxes_nonzero = torch.nonzero(torch.sum(output_boxes, dim=-1), as_tuple=True)[0]
            output_boxes = output_boxes[target_boxes_nonzero]

            loss_iou += self.iou_loss(output_boxes, target_boxes, fsize=fsize)

        loss = loss_obj + loss_class + loss_iou
        return loss, loss_obj, loss_class, loss_iou


def train_one_epoch(model, data_loader, criteria, optimizer, scheduler=None, device=None, epoch=0):
    '''
    train model one epoch
    '''
    # model.to(device)
    # criteria = Yolo_loss(device=device, batch_size=batchsize)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    global_step = 0
    losses, losses_obj, losses_class, losses_iou = 0, 0, 0, 0
    model.train()
    for img, target in data_loader:
        img = img.to(device)
        if isinstance(target, dict):
            target = {key: target[key].to(device) for key in target.keys()}
        else:
            target = target.to(device)
        pred = model(img)
        loss, loss_obj, loss_class, loss_iou = criteria(pred, target)
        
        losses += loss
        losses_obj += losses_obj
        losses_class += losses_class
        losses_iou += losses_iou
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()
        global_step = global_step + batchsize
        if global_step % 200 == 0:
            print("{}/{} :loss {:5.4f} loss_obj {:5.4f} loss_class {:5.4f} loss_iou {:5.4f}".format(epoch,
                                                                                                    global_step,
                                                                                                    loss/batchsize,
                                                                                                    loss_obj/batchsize,
                                                                                                    loss_class/batchsize,
                                                                                                    loss_iou/batchsize))
    # torch.save
    return losses, losses_obj, losses_class, losses_iou
    


def train(model, data_loader, batchsize=4, epoch=30, device=None):
    '''
    train model
    '''
    device_ids = [4, 5, 6, 7]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    model.train()
    def burnin_scheduler(epoch):
        factor = 1.0
        if epoch < 10:
            factor = 1.0
        elif epoch < 20:
            factor = 0.5
        elif epoch < 30:
            factor = 0.1
        return factor

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_scheduler, last_epoch=-1)
    criteria = Yolo_loss(device=device, batch_size=batchsize)    

    data_size = len(data_loader.dataset)
    for e in range(epoch):
        print("start training epoch: {}".format(e))
        losses, losses_obj, losses_class, losses_iou = train_one_epoch(model, 
                                                                       data_loader, 
                                                                       criteria, 
                                                                       optimizer, 
                                                                       scheduler, 
                                                                       device, 
                                                                       e)
        print("epoch {}: losses: {:5.4} losses_obj: {:5.4} losses_class: {:5.4} losses_iou: {:5.4}".format(e,
													                                                       losses/data_size,
                                                                                                           losses_obj/data_size,
                                                                                                           losses_class/data_size,
                                                                                                           losses_iou/data_size))
        

        
    

if __name__ == "__main__":

    batchsize = 16
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    model = DarkNet('./cfg/yolov4.cfg', True)
    model.load_weight('./data/yolov4.weights')
    model = model.to(device=device)
    dataset = Coco_dataset('/root/data/cocodataset', train=True)
    dataloader = DataLoader(dataset, batch_size=batchsize, collate_fn=yolo_collate_fn, shuffle=True, drop_last=True)
    print("dataloader size : {}".format(len(dataloader.dataset)))

    # img, target = next(iter(dataloader))
    # # pprint(img)
    # img = img.to(torch.float).to(device)

    # outfeatures = model(img)

    # yololoss = Yolo_loss(device=device, batch_size=batchsize)
    # loss, loss_obj, loss_class, loss_iou = yololoss(outfeatures, target)
    # print('loss: ', loss.cpu().detach().numpy())
    # print('loss_obj: ', loss_obj.cpu().detach().numpy())
    # print('loss_class: ', loss_class.cpu().detach().numpy())
    # print('loss_iou: ', loss_iou.cpu().detach().numpy())

    train(model, dataloader, batchsize=batchsize, device=device)
