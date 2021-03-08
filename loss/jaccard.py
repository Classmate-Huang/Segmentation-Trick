'''
  Reference : https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/jaccard.py
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F 


BINARY_MODE='binary'    # 二分类任务
MULTICLASS_MODE='multiclass'    # 多分类任务
MULTILABEL_MODE='multilabel'    # 多标签任务(同一像素可属于多个标签)


def to_tensor(x, dtype=None):
    ''' convert to Tensor '''
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

def soft_jaccard_score(output, target, smooth, eps, dims):
    ''' implement soft Jaccard/IoU score 
    output : Tensor [N, C, HxW]
    target : Tensor [N, C, HxW]
    '''
    assert output.size()==target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims) # [C]
        cardinality = torch.sum(output + target, dim=dims)  # [C]
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    
    union = cardinality - intersection
    jaccard_score = (intersection + smooth)/(union + smooth).clamp_min(eps)

    return jaccard_score


class JaccardLoss(nn.Module):
    
    def __init__(self, mode='multiclass', classes=None, log_loss=None, 
                from_logits=True, smooth=0., eps=1e-7):
        ''' Jaccard / IoU Loss '''
        assert mode in {BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE}
        super(JaccardLoss, self).__init__()

        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth 
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, pred, target):

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                pred = pred.log_softmax(dim=1).exp()    # softmax
            else:
                pred = F.logsigmoid(pred).exp()     # logits

        batchsize = target.size(0)
        num_class = pred.size(1)
        dims = (0,2)

        if self.mode == BINARY_MODE:
            target = target.view(batchsize, 1, -1)
            pred = pred.view(batchsize, 1, -1)

        if self.mode == MULTICLASS_MODE:
            target = target.view(batchsize, -1) # [N, HxW]
            pred = pred.view(batchsize, num_class, -1)  # [N, C, HxW]

            target = F.one_hot(target, num_class)   # [N, HxW, C]
            target = target.permute(0, 2, 1)    # [N, C, HxW]
        
        if self.mode == MULTILABEL_MODE:
            target = target.view(batchsize, num_class, -1)
            pred = pred.view(batchsize, num_class, -1)

        score = soft_jaccard_score(pred, target, smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(score.clamp_min(self.eps))
        else:
            loss = 1.0 - score

        # zero contribution of channel that doesn't have true pixels        
        mask = target.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


if __name__ == '__main__':

    x = torch.randn(3,2,4,5).type(torch.float32)
    y = torch.randn(3,2,4,5).type(torch.long)

    criterion = JaccardLoss(mode='multiclass', from_logits=True)
    loss = criterion(x, y)
    print(loss.item())