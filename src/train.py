import os 
import os.path as osp
import numpy as np 
import time 
from tqdm import tqdm

from models.scenegraphmerger import *
from common.base import *
from common.logger import colorlogger
from common.timer import Timer
from loss import *
from datasets import ssg3D
import settings as CONF

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ModelTrainer(Trainer):
    def __init__(self):
        super(ModelTrainer, self).__init__()
        self.zoom = CONF.ZOOM
        

    def innerViewLoss(self, embs, dataDict):
        losses = []
        for trainMode in self.trainMode:
            loss = self.criterionCl(embs[trainMode], dataDict)
            losses.append(loss)
        
        if len(losses) > 1:
            totalLoss =  self.multiLossLayer(losses)
            return totalLoss
        else:
            return losses[0]
    
    def klAlignmentLoss(self, embs, dataDict):
        assert len(self.trainMode) > 1

        losses = []
        for trainMode in self.trainMode:
            loss = self.criterionAlign(embs[trainMode], embs['joint'], dataDict)
            losses.append(loss)
        totalLoss = self.alignMultiLossLayer(losses) * self.zoom
        return totalLoss
    
    def train(self):
        for epoch in range(self.startEpoch, self.endEpoch):
            self.model.train()
            if len(self.trainMode) > 1:
                self.multiLossLayer.train()
                self.alignMultiLossLayer.train()
            
            self.tot_timer.tic()
            self.read_timer.tic()
            
            # Training
            lossMeter = {'icl_uni_modal' : AverageMeter()}
            if len(self.trainMode) > 1:
                lossMeter['icl_joint'] = AverageMeter()
                lossMeter['ial_align'] = AverageMeter()

            for iter, dataDict in enumerate(self.trainDataLoader):
                loss = {}
                for key in dataDict:
                    if key not in CONF.NP_KEYNAMES:
                        dataDict[key] = dataDict[key].to(self.device)
                dataDict['batch_size'] = self.batchSize

                self.read_timer.toc()
                self.gpu_timer.tic()

                # forward 
                self.optimizer.zero_grad()                
                embs = self.model(dataDict)

                loss['icl_uni_modal'] = self.innerViewLoss(embs, dataDict)

                if len(self.trainMode) > 1:
                    loss['icl_joint'] = self.criterionCl(embs['joint'], dataDict)
                    loss['ial_align'] = self.klAlignmentLoss(embs, dataDict) # IAL loss for uni-modal embedding
                    lossMeter['icl_joint'].update(loss['icl_joint'], dataDict['tot_obj_count'].shape[0])
                    lossMeter['ial_align'].update(loss['ial_align'], dataDict['tot_obj_count'].shape[0])

                if iter % 100 == 0:
                    self.writer.add_scalars('train',  
                            loss
                            , epoch * len(self.trainDataLoader) + iter
                    )
                # backward
                sum(loss[k] for k in loss).backward()
                self.optimizer.step()
                self.gpu_timer.toc()

                if iter % 1000 == 0:
                    screen = [
                    '[INFO] Training', 
                    'Train Mode : {}'.format('_'.join(self.trainMode)),
                    'Train Epoch %d/%d iter  %d/%d:' % (epoch, self.endEpoch, iter+1, self.trainIterPerEpoch),
                    'lr: %g' % (self.lr)
                    ]
                    screen += ['%s: %.4f' % ('loss_' + k, v.avg) for k,v in lossMeter.items()]
                    self.logger.info(' '.join(screen))

                self.tot_timer.toc()
                self.tot_timer.tic()
                self.read_timer.tic()

            self.saveModel({
                'epoch' : epoch,
                'network' : self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict()
            }, epoch)


if __name__ == '__main__':
    trainerObj = ModelTrainer()
    trainerObj._makeBatchGenerator(split='train')
    # trainerObj._makeBatchGenerator(split='val')
    trainerObj._makeModel(training=True)
    
    trainerObj.train()
