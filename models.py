import os 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

from collections import OrderedDict
from pprint import pprint
import cv2

from cocodataset import save_image_boxes, load_class_names

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape))
    start = start + num_w
    return start

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape))
    start = start + num_w
    return start

def get_iou(b1, b2, iou_type='iou'):
    '''
    boxes must be xyxy mode tensor
    boxes shape: (n, 4) (m, 4)
    return shape (n, m)
    '''
    # device = b1.device
    # 0. get width and height and center coordinate
    b1_w = b1[..., 2] - b1[..., 0]
    b1_h = b1[..., 3] - b1[..., 1]
    b2_w = b2[..., 2] - b2[..., 0]
    b2_h = b2[..., 3] - b2[..., 1]

    b1_center_x = (b1[..., 2] + b1[..., 0]) / 2
    b1_center_y = (b1[..., 3] + b1[..., 1]) / 2
    b2_center_x = (b2[..., 2] + b2[..., 0]) / 2
    b2_center_y = (b2[..., 3] + b2[..., 1]) / 2

    b1 = b1.view(b1.shape[0], 1, b1.shape[1])

    # 1. get inter boxes
    left = torch.max(b1[..., 0], b2[..., 0]).unsqueeze(2)
    top = torch.max(b1[..., 1], b2[..., 1]).unsqueeze(2)
    right = torch.min(b1[..., 2], b2[..., 2]).unsqueeze(2)
    bottom = torch.min(b1[..., 3], b2[..., 3]).unsqueeze(2)
    inter_boxes = torch.cat((left, top, right, bottom), dim=2)

    # 2. get convex boxes
    conv_topleft = torch.min(b1[..., :2], b2[..., :2])
    conv_botright = torch.max(b1[..., 2:], b2[..., 2:])
    # conv_boxes = torch.cat((conv_topleft, conv_botright), dim=2)

    # 3. get areas
    inter_areas = (inter_boxes[..., 2] - inter_boxes[..., 0]).clamp(min=0) * (inter_boxes[..., 3] - inter_boxes[..., 1]).clamp(min=0)
    # conv_areas = (conv_boxes[..., 2] - conv_boxes[..., 0]).clamp(min=0) * (conv_boxes[..., 3] - conv_boxes[..., 1]).clamp(min=0)
    conv_areas = torch.prod((conv_botright - conv_topleft), dim=2, keepdim=False)
    # inter_areas = inter_areas.clamp(min=0)

    b1_area = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    b2_area = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_areas = b1_area + b2_area - inter_areas

    # 4. get center distance
    center_distance = ((b1[..., 2] + b1[..., 0]) - (b2[..., 2] + b2[..., 0])) ** 2 / 4 + \
                      ((b1[..., 3] + b1[..., 1]) - (b2[..., 3] + b2[..., 1])) ** 2 / 4

    if (iou_type == 'iou'):
        ious = inter_areas / union_areas
    elif (iou_type == 'giou'):
        ious = (inter_areas / union_areas) - ((conv_areas - union_areas) / conv_areas)
    elif (iou_type == 'diou' or iou_type == 'ciou'):
        conv_distances = torch.pow(conv_botright - conv_topleft, 2).sum(dim=2) + 1e-16
        if (iou_type == 'diou'):
            ious = (inter_areas / union_areas) - (center_distance / conv_distances)
        elif (iou_type == 'ciou'):
            v = (4 / np.pi ** 2) * torch.pow(torch.atan(b1_w / b1_h).unsqueeze(1) - torch.atan(b2_w / b2_h), 2)
            with torch.no_grad():
                alpha = v / (1 - (inter_areas / union_areas) + v + 1e-16)
            ious = (inter_areas / union_areas) - (center_distance / conv_distances + v * alpha)
            ious = torch.clamp(ious, min=-1.0, max=1.0)
    
    return ious

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride
    def forward(self, x):
        assert(x.data.dim() == 4)
        B, C, H, W = x.shape
        x = x.view(B, C, H, 1, W, 1)
        x = x.expand(B, C, H, self.stride, W, self.stride).contiguous()
        x = x.view(B, C, H*self.stride, W*self.stride)
        return x

class Upsample_interpolation(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_interpolation, self).__init__()
        self.stride = stride 
    def forward(self, x):
        assert(x.data.dim() == 4)

        out = F.interpolate(x, size=(x.size(2)*self.stride, x.size(3)*self.stride), mode='nearest')
        return out

class Route_layer(nn.Module):
    def __init__(self, layers):
        super(Route_layer, self).__init__()
        self.layers = layers

    def forward(self, x):
        return x

class Shortcut_layer(nn.Module):
    def __init__(self, from_):
        super(Shortcut_layer, self).__init__()
        self.from_ = from_

    def forward(self, x):
        return x

class Yolo_layer(nn.Module):
    def __init__(self, anchors, stride=8):
        super(Yolo_layer, self).__init__()
        self.anchors = anchors
        self.stride = stride

    def forward(self, x, is_train=True):
        if is_train:
            return x 
        # for inference
        # 1. transpose to B x predictions x (num_classes+5)
        self.anchors = self.anchors.to(x.device)
        num_anchors = self.anchors.shape[0]
        num_classes = (x.shape[1] // num_anchors) - 5
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        num_channels = x.shape[1]
        x = x.view(batch_size, num_channels, height*width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, height*width*num_anchors, num_classes+5)
        # 2. sigmoid to x classes' pro and obj pro
        x[:, :, [0,1]] = torch.sigmoid(x[:, :, [0,1]])
        x[:, :, 4:] = torch.sigmoid(x[:, :, 4:])
        # 3. add offset to bboxes 
        range_x = np.array(range(width), dtype=np.float)
        range_y = np.array(range(height), dtype=np.float)
        offset_x, offset_y = np.meshgrid(range_x, range_y)
        offset_x = np.repeat(offset_x[:, :, np.newaxis], num_anchors, 2).flatten()[:, np.newaxis]
        offset_y = np.repeat(offset_y[:, :, np.newaxis], num_anchors, 2).flatten()[:, np.newaxis]
        offset_xy = np.concatenate((offset_x, offset_y), axis=1)
        offset_xy = torch.from_numpy(offset_xy).to(torch.float).to(x.device)
        x[:, :, [0,1]] += offset_xy
        # 4. bboxes to real size and from c_x c_y w h to x1 y1 x2 y2
        fea_anchors = self.anchors / self.stride
        fea_anchors = fea_anchors.repeat(height*width, 1)
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * fea_anchors

        x[:, :, 0:4] *= self.stride
        x[:, :, 0:2] -= x[:, :, 2:4] / 2
        x[:, :, 2:4] += x[:, :, 0:2]
        # 5. get classes
        class_pro, class_id = torch.max(x[:, :, 5:], dim=2, keepdim=True)
        predictions = torch.cat((x[:, :, :5], class_pro, class_id), dim=2)

        # predictions shape: B x (H x W x num_anchors) x 7
        # 7 = rect_x rect_y rect_w rect_h obj_pro cls_pro cls_id
        # print(predictions.shape)
        return predictions

def get_model_from_cfg(filename):
    '''
    read cfg file to get model dict
    '''
    model_list = []
    model_dict = {}
    model_set = set()
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if (len(line) <= 0 or line[0] == '#'):
                continue
            # print(line)
            if (line[0] == '['):
                if bool(model_dict):
                    model_list.append(model_dict)
                model_dict = {}
                model_dict['property'] = line[1:-1]
            else:
                attrs = line.split('=')
                model_dict[attrs[0].strip()] = attrs[1].strip()
                if model_dict['property'] == 'shortcut':
                    if attrs[0] == 'activation':
                        model_set.add(attrs[1])
        model_list.append(model_dict)
    for item in model_set:
        print(item)
    return model_list

def create_network(blocks):
    if not bool(blocks):
        print("error: blocks is NULL")
        raise KeyError
    models = nn.ModuleList()
    pre_filters = 3
    out_filters = []
    pre_stride = 1
    out_strides = []
    info = {}
    conv_id = 0
    for block in blocks:
        if block['property'] == 'net':
            pre_filters = int(block['channels'])
        elif block['property'] == 'convolutional':
            conv_id += 1
            batch_norm = 'batch_normalize' in block
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            activation = block['activation']
            pad = (kernel_size - 1) // 2 if is_pad else 0
            net = nn.Sequential()
            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                conv = nn.Conv2d(pre_filters, filters, kernel_size, stride, pad, bias=False)
                net.add_module('conv_{0}'.format(conv_id), conv)
                net.add_module('bn_{0}'.format(conv_id), bn)
            else:
                conv = nn.Conv2d(pre_filters, filters, kernel_size, stride, pad)
                net.add_module('conv_{0}'.format(conv_id), conv)

            if activation == 'mish':
                act = Mish()
                net.add_module('mish_{0}'.format(conv_id), act)
            elif activation == 'leaky':
                act = nn.LeakyReLU(0.1, inplace=True)
                net.add_module('leaky_{0}'.format(conv_id), act)
            elif activation == 'relu':
                act = nn.ReLU(inplace=True)
                net.add_module('relu_{0}'.format(conv_id), act)
            else:
                print("convolutional {0} doesn't have activation".format(conv_id))

            pre_filters = filters
            out_filters.append(pre_filters)
            pre_stride = stride * pre_stride
            out_strides.append(pre_stride)
            models.append(net)
        
        elif block['property'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            pad = (pool_size - 1) // 2
            models.append(nn.MaxPool2d(pool_size, stride, pad))

            out_filters.append(pre_filters)
            pre_stride = stride * pre_stride
            out_strides.append(pre_stride)

        elif block['property'] == 'upsample':
            stride = int(block['stride'])
            models.append(Upsample_expand(stride))

            out_filters.append(pre_filters)
            pre_stride = pre_stride // stride
            out_strides.append(pre_stride)

        elif block['property'] == 'route':
            layers = block['layers']
            layers = [int(layer) for layer in layers.split(',')]
            models.append(Route_layer(layers))

            pre_filters = sum([out_filters[layer] for layer in layers])
            pre_stride = out_strides[layers[0]]

            out_filters.append(pre_filters)
            out_strides.append(pre_stride)

        elif block['property'] == 'shortcut':
            from_ = int(block['from'])
            activation = block['activation']

            if activation == 'linear':
                models.append(Shortcut_layer(from_))

            out_filters.append(pre_filters)
            out_strides.append(pre_stride)

        elif block['property'] == 'yolo':
            mask = [int(s) for s in block['mask'].split(',')]
            anchors = [int(s) for s in block['anchors'].split(',')]
            anchors = [[anchors[i*2], anchors[i*2+1]] for i in range(len(anchors)//2)]
            anchors = np.array(anchors)
            anchors = torch.from_numpy(anchors).to(torch.float)[mask]
            scale_x_y = float(block['scale_x_y'])
            ignore_thresh = float(block['ignore_thresh'])
            num_classes = int(block['classes'])
            models.append(Yolo_layer(anchors, pre_stride))

            out_filters.append(pre_filters)
            out_strides.append(pre_stride)

        else:
            print("{} not found".format(block['property']))

    return models

def post_process(yolo_out, obj_threshold=0.5, iou_threshold=0.5):
    '''
    get rid of items below obj_threshold and do nms
    yolo_out: tensor (B, predictions, 7)
    output: list which size is B, each element is predictions of that img,
            if there is none prediction in that img, the element is tensor([])
    '''
    # help act please delete after having write
    # yolo_out = torch.tensor(yolo_out)
    # done 

    def get_tensor_unique(t, axis=None):
        device = t.device
        np_t = t.cpu().numpy()
        np_uni = np.unique(np_t, axis=axis)
        t_uni = torch.from_numpy(np_uni).to(torch.float).to(device)
        return t_uni
    
    batchs_out = []
    batch_size = yolo_out.shape[0]
    for b in range(batch_size):
        predictions = yolo_out[b]
        # 1. get rid of objs whose obj score below obj_threshold
        # pprint(predictions.shape)
        obj_idx = predictions[:, 4] > obj_threshold
        predictions = predictions[obj_idx]
        if predictions.shape[0] == 0:
            batchs_out.append(torch.empty([]))
        # pprint(predictions)

        # 2. do nms
        classes = get_tensor_unique(predictions[:, -1])
        for class_id in classes:
            predictions_mask = predictions[:, -1] == class_id
            predictions_c = predictions[predictions_mask]
            _, ind = predictions_c[:, 4].sort(dim=0, descending=True)
            predictions_c = predictions_c[ind]
            start_id = 1
            num_class = predictions_c.shape[0]
            for idx in range(num_class):
                if start_id >= predictions_c.shape[0]:
                    break
                ious = get_iou(predictions_c[start_id-1:start_id, :4], predictions_c[start_id:, :4], iou_type='ciou')
                ious_mask = ious.squeeze() < iou_threshold
                predictions_c[start_id:, -1] *= ious_mask
                valid_idx = torch.nonzero(predictions_c[start_id:, -1], as_tuple=True)[0]
                start_id += valid_idx[0]+1 if valid_idx.shape[0] > 0 else predictions_c.shape[0]
            # pprint(predictions_c)
            predictions[predictions_mask] = predictions_c

        nonzero_ids = torch.nonzero(predictions[:, -1], as_tuple=True)[0]
        predictions = predictions[nonzero_ids]
        batchs_out.append(predictions)

    return batchs_out

class DarkNet(nn.Module):
    def __init__(self, cfgfile, is_train=True):
        super(DarkNet, self).__init__()
        self.is_train = is_train
        self.blocks = get_model_from_cfg(cfgfile)
        self.model_list = create_network(self.blocks)
        self.anchors = []
        self.anchor_masks = []
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x):
        
        out_features = []
        outputs = []
        for ind, block in enumerate(self.blocks):
            if block['property'] == 'net':
                continue
            
            elif block['property'] == 'convolutional':
                module = self.model_list[ind-1]
                x = module(x)
                out_features.append(x)

            elif block['property'] == 'maxpool':
                module = self.model_list[ind-1]
                x = module(x)
                out_features.append(x)

            elif block['property'] == 'shortcut':
                from_ = int(block['from'])
                x = x + out_features[from_]
                out_features.append(x)

            elif block['property'] == 'route':
                layers = block['layers']
                layers = [int(layer) for layer in layers.split(',')]
                if len(layers) == 1:
                    x = out_features[layers[0]]
                elif len(layers) == 2:
                    x1 = out_features[layers[0]]
                    x2 = out_features[layers[1]]
                    x = torch.cat((x1, x2), dim=1)
                elif len(layers) == 4:
                    x1 = out_features[layers[0]]
                    x2 = out_features[layers[1]]
                    x3 = out_features[layers[2]]
                    x4 = out_features[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), dim=1)
                out_features.append(x)
            
            elif block['property'] == 'upsample':
                module = self.model_list[ind-1]
                x = module(x)
                out_features.append(x)

            elif block['property'] == 'yolo':
                module = self.model_list[ind-1]
                x = module(x, self.is_train)
                out_features.append(x)
                outputs.append(x)
        if not self.is_train:
            result = torch.cat(outputs, dim=1)
        else:
            result = outputs
        return result

    def get_anchors(self):
        return self.anchors, self.anchor_masks

    def load_weight(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0

        for i, block in enumerate(self.blocks):

            if block['property'] == 'convolutional':
                model = self.model_list[i-1]
                if 'batch_normalize' in block:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])

            else:
                pass


                    

if __name__ == "__main__":
    # model_blocks = get_model_from_cfg('./cfg/yolov4.cfg')
    # net = create_network(model_blocks)
    # for item in model_blocks:
    #     pprint(item)
    # pprint(net)

    # x = torch.randn((16, 3, 608, 608))
    # model_train = DarkNet('./cfg/yolov4.cfg')
    # result = model_train(x)
    # pprint(type(result))
    # pprint(len(result))
    # pprint([r.shape for r in result])
    print('-'*55)
    b1 = torch.tensor([[0,0,2,2], [0,0,4,4]]).to(torch.float)
    b2 = torch.tensor([[0,0,2,2], [0,0,2,4], [1,1,4,4],[4,5,5,6]]).to(torch.float)
    ious = get_iou(b1, b2, iou_type='diou')
    pprint(ious)

    model_val = DarkNet('./cfg/yolov4.cfg', False)
    model_val.load_weight('./data/yolov4.weights')
    # pprint(model_val.get_anchors())
    # val_result = model_val(x)
    # pprint(type(val_result))
    # pprint(val_result.shape)
    with torch.no_grad():

        model_val.eval()
        img = cv2.imread('./data/dog.jpg')
        img = cv2.resize(img, (608, 608), interpolation=cv2.INTER_CUBIC)
        new_img = img.copy()
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, :, :, :] / 255.0
        img = torch.tensor(img).to(torch.float)
        result = model_val(img)
        result = post_process(result)
        boxes = result[0]
        pprint(boxes.shape)
        num_boxes = 0
        bboxes = []
        classes = []
        classes_name = load_class_names('./data/coco.names')
        for box in boxes:
            # if box[4] > 0.5:
            num_boxes += 1
            rect = box[:4]
            bboxes.append(rect.unsqueeze(0))
            classes.append(classes_name[box[-1].to(torch.int)])

        bboxes = torch.cat(bboxes, dim=0)
        # pprint(boxes[boxes[:, 4]>0.5])
        
        print("box num is {}: ".format(num_boxes))
        save_image_boxes(new_img, bboxes, 'imgpredicboxes.jpg', classes)

        # post = post_process(pre_result)

        # b1 = torch.tensor([[0,0,2,2], [0,0,4,4]]).to(torch.float)
        # b2 = torch.tensor([[0,0,2,2], [0,0,2,4], [1,1,4,4],[4,5,5,6]]).to(torch.float)
        # ious = get_iou(b1, b2)
        # pprint(ious)
    