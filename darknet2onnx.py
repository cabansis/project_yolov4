import torch 
from models import DarkNet 

def transform_darknet_2_onnx(cfgfile, weightfile, batchsize=1, height=608, width=608):
    '''
    transform darknet model to onns
    '''
    model = DarkNet(cfgfile, is_train=False)
    model.load_weight(weightfile)
    print('...load weights from %s' % (weightfile))
    
    input_name = ['input']
    output_name = ['output']

    if (batchsize > 0):
        x = torch.randn((batchsize, 3, height, width), requires_grad=True)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batchsize, height, width)
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          verbose=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_name,
                          output_names=output_name,
                          dynamic_axes=None)
        return onnx_file_name
    else:
        x = torch.randn((1, 3, height, width), requires_grad=True)
        onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(height, width)
        dynamic_axes = {"input":{0: "batchsize"}, "output": {0: "batchsize"}}
        print("Export the onnx model ...")
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          verbose=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_name,
                          output_names=output_name,
                          dynamic_axes=dynamic_axes)
        print("Onnx model exporting done ...")
        return onnx_file_name

if __name__ == "__main__":
    cfgfile = '/home/baodi/yolo_projects/project_yolov4/cfg/yolov4.cfg'
    weightfile = '/home/baodi/yolo_projects/project_yolov4/data/yolov4.weights'

    transform_darknet_2_onnx(cfgfile, weightfile, batchsize=-1)