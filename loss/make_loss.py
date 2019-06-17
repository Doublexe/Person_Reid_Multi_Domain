# encoding: utf-8


import torch.nn.functional as F

from .triplet_loss import TripletLoss
from .imptriplet_loss import ImpTripletLoss

def make_loss(cfg):
    loss_selector = cfg.SOLVER.LOSS
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    imptriplet = ImpTripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    
    if loss_selector == 'softmax':
        def loss_func(score, feat, labels):
            return F.cross_entropy(score, labels)
    elif loss_selector == 'triplet':
        def loss_func(score, feat, labels):
            return triplet(feat, labels)[0]
    elif loss_selector == 'softmax_triplet':
        def loss_func(score, feat, labels):
            return F.cross_entropy(score, labels) + triplet(feat, labels)[0]
    elif loss_selector == 'softmax_imptriplet':
        def loss_func(score, feat, labels):
            return F.cross_entropy(score, labels) + imptriplet(feat, labels)[0]
    elif loss_selector == 'MGN':
        def loss_func(scores, feats, labels):
            T_Loss = [triplet(feat, labels)[0] for feat in feats]
            T_Loss = sum(T_Loss) / len(T_Loss)
            C_Loss = [F.cross_entropy(score, labels) for score in scores]
            C_Loss = sum(C_Loss) / len(C_Loss)
            return T_Loss + 2 * C_Loss
    else:
        print('expected loss_selector should be softmax, triplet, softmax_triplet or softmax_imptriplet, ''but got {}'.format(loss_selector))
        
    return loss_func
