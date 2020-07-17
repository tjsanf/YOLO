import time
import random
import threading
import cv2
import numpy as np

class batch_thread (threading.Thread):
    ready = False
    batch_data_length = 0
    batch_data_list = []
    min_data_size = 1
    max_data_size = 10

    def __init__(self, jpg_list, label_list, batch_size, min_data_size = 1, max_data_size = 60, end = False):
        threading.Thread.__init__(self)
        self.jpg_list = jpg_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.min_data_size = min_data_size
        self.max_data_size = max_data_size
        self.total_length = len(self.jpg_list)
        self.total_indexs = np.arange(self.total_length).tolist()
        self.total_iteration = int(self.total_length/self.batch_size)

    def get_batch_data(self):
        batch_image_data, batch_label_data = self.batch_data_list[0]

        del self.batch_data_list[0]
        self.batch_data_length -= 1

        if self.batch_data_length < self.min_data_size:
            self.ready = False

        return batch_image_data, batch_label_data

    def run(self):
        while True:
            while self.batch_data_length >= self.max_data_size:
                time.sleep(0.01)
                continue

            batch_image_data = []
            batch_label_data = []
            batch_indexs = random.sample(self.total_indexs, self.batch_size)

            _jpg_list = self.jpg_list[batch_indexs]
            _label_list = self.label_list[batch_indexs]
            for jpg_name, xml_name in zip(_jpg_list, _label_list):
                _img = cv2.imread("./VOCdevkit_train/"+jpg_name, cv2.IMREAD_COLOR)
                _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
                _gtd = annotation(xml_name)
                _img, _bb, _cl = da.DataAugmentation(_img, _gtd[:,:4], _gtd[:,4])
                _gtd = makegtd(_bb, _cl, np.shape(_img)[0], np.shape(_img)[1])
                _img = cv2.resize(_img, dsize=(Is, Is), interpolation=cv2.INTER_AREA)
                batch_image_data.append(_img); batch_label_data.append(_gtd)

            batch_image_data = np.array(batch_image_data)
            batch_label_data = np.array(batch_label_data)

            self.batch_data_list.append([batch_image_data, batch_label_data])
            self.batch_data_length += 1

            if self.batch_data_length >= self.min_data_size:
                self.ready = True
            else:
                self.ready = False