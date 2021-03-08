''' calculate Mean IOU '''
import numpy as np


class SegMetric(object):
    
    def __init__(self, n_class):
        self.n_class = n_class
        self.confusion_matrix = np.zeros((self.n_class, self.n_class))
    
    def generate_confusion_matrix(self, preds, labels):
        '''generate confusion matrix from prediction and labels'''
        index = (labels>=0) & (labels<self.n_class)
        mask = self.n_class * labels[index] + preds[index]
        count = np.bincount(mask, minlength=self.n_class**2)
        confusion_matrix = count.reshape(self.n_class, self.n_class)
        return confusion_matrix
    
    def add_batch(self, preds, labels):
        ''' update confusion matrix every batch '''
        assert preds.shape == labels.shape
        self.confusion_matrix += self.generate_confusion_matrix(preds, labels)

    def reset(self):
        ''' reset confusion matrix '''
        self.confusion_matrix = np.zeros((self.n_class, self.n_class))

    def pixel_accuracy(self):
        ''' calculate pixel accuracy '''
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_class_accuracy(self):
        ''' calculate pixel accuracy per class '''
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return acc

    def mean_class_accuracy(self):
        ''' calculate mean pixel accuracy of all class  '''
        acc = np.nanmean(self.pixel_class_accuracy)
        return acc
    
    def iou(self):
        I = np.diag(self.confusion_matrix)
        U = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
        IoU = I / U
        meanIoU = np.nanmean(IoU)
        return IoU, meanIoU
    

if __name__ == '__main__':
    imgPredict = np.array([0, 1, 1, 1, 2, 2])
    imgLabel = np.array([0, 0, 1, 1, 2, 2])
    metric = SegMetric(n_class=3)
    metric.add_batch(imgPredict, imgLabel)
    acc = metric.pixel_accuracy()
    IoU, mIoU = metric.iou()
    metric.reset()
    print(acc, IoU, mIoU)
