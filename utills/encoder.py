import xml.etree.ElementTree as elemTree
import numpy as np

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

def makegtd(bonding_box, classes, hight, width, grid, model):
    _val = np.zeros([grid,grid,2,25])
    for bb, _c in zip(bonding_box, classes):
        _x = (bb[0] + bb[2])/2/width*grid
        _y = (bb[1] + bb[3])/2/hight*grid
        _w = (bb[2] - bb[0])/width
        _h = (bb[3] - bb[1])/hight
        _vx = _x%1; _px = int(_x); _vy = _y%1; _py = int(_y)

        if model == 'yolo_v1':

        if _val[_py, _px, 0, 0] == 0:
            _val[_py, _px, :, 0] = _vx; _val[_py, _px, :, 1] = _vy; _val[_py, _px, :, 2] = _w; _val[_py, _px, :, 3] = _h; _val[_py, _px, :, 4] = 1; _val[_py, _px, :, int(_c)] = 1
        else:
            _val[_py, _px, 1, :] = 0
            _val[_py, _px, 1, 0] = _vx; _val[_py, _px, 1, 1] = _vy; _val[_py, _px, 1, 2] = _w; _val[_py, _px, 1, 3] = _h; _val[_py, _px, 1, 4] = 1; _val[_py, _px, 1, int(_c)] = 1

    return _val