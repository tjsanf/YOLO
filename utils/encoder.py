import random
import numpy as np
import xml.etree.ElementTree as elemTree


def annotation(xml_name):
    '''
    xml to list
    inpout xml_path
    output [?, 25]
    '''
    cat = {"person" : 5, "bird" : 6, "cat" : 7, "cow" : 8, "dog" : 9, "horse" : 10, "sheep" : 11, "aeroplane" : 12, "bicycle" : 13, "boat" : 14, "bus" : 15, "car" : 16, "motorbike" : 17, "train" : 18, "bottle" : 19, "chair" : 20, "diningtable" : 21, "pottedplant" : 22, "sofa" :23, "tvmonitor" : 24}
    tree = elemTree.parse('./VOCdevkit_train/'+xml_name)

    val = []
    for obj in tree.findall('./object'):
        sub_val = [float(obj.find('bndbox/xmin').text), float(obj.find('bndbox/ymin').text), float(obj.find('bndbox/xmax').text), float(obj.find('bndbox/ymax').text), cat[obj.find('name').text]]
        val.append(sub_val)

    return np.asarray(val)

def makegtd(bonding_box, classes, hight, width, output_data, model):
    if model == 'yolo_v1':
        _, w, h, g = output_data.shape
        _val = np.zeros([w, h, 2, 25])
    elif model == 'yolo_v2':
        _, w, h, x, g = output_data.shape
        _val = np.zeros([w, h, x, g])

    for bb, _c in zip(bonding_box, classes):
        _x = (bb[0] + bb[2])/2/width*w
        _y = (bb[1] + bb[3])/2/hight*h
        _w = (bb[2] - bb[0])/width
        _h = (bb[3] - bb[1])/hight
        _vx = _x%1; _px = int(_x); _vy = _y%1; _py = int(_y)

        if _val[_py, _px, 0, 0] == 0:
            for i in _val.shape[2]:
                _val[_py, _px, i, 0] = _vx; _val[_py, _px, i, 1] = _vy; _val[_py, _px, i, 2] = _w; _val[_py, _px, i, 3] = _h; _val[_py, _px, i, 4] = 1; _val[_py, _px, i, int(_c)] = 1
        else:
            for i in _val.shape[2]:
                if random.random() > 0.5:
                    _val[_py, _px, i, :] = 0
                    _val[_py, _px, i, 0] = _vx; _val[_py, _px, i, 1] = _vy; _val[_py, _px, i, 2] = _w; _val[_py, _px, i, 3] = _h; _val[_py, _px, i, 4] = 1; _val[_py, _px, i, int(_c)] = 1
        
    if model == 'yolo_v1':
        val = np.zeros([w, h, g])
        val[:,:,:5] = _val[:,:,:5]
        val[:,:,5:10] =  _val[:,:,:5]
        val[:,:,10:] = _val[:,:,5:]
        _val = val

    return _val