import torch
import torch.nn as nn
from config import Constants
from models.Predictor.pred_attribute import (
    get_prj_by_flag, 
    prepare_merged_probs
)
from .base import CritBase
from ..logger import AverageMeter


class NoisyOrMIL(CritBase):
    def __init__(self, opt, keys=None):
        super().__init__(
            keys=['preds_attr', 'avg_prob_attr', 'labels_attr'] if keys is None else keys,
            batch_mean=True
        )
        self.topk_list = [10, 20, 30, 40, 50]
        self.calculate_mAP = opt.get('calculate_mAP', False)
        self.regularization_loss_fn = nn.L1Loss(reduction='none')

    def _step(self, index_indicator, preds_attr, avg_prob_attr, labels_attr, *others):
        """
            args:
                preds_attr:     (batch_size, n_attributes)
                avg_prob_attr:  (batch_size, )
                labels_attr:    (batch_size, n_attributes), multi-hot
        """
        assert not len(others)
        assert preds_attr.shape[1] <= labels_attr.shape[1]
        preds_attr = torch.clamp(preds_attr, 0.01, 0.99) # avoid nan
        labels_attr = labels_attr[:, :preds_attr.shape[1]].to(preds_attr.device)
        
        device = preds_attr.device
        n_positive = labels_attr.sum(1).float()
        n_attributes = preds_attr.shape[1]
        
        # bce loss normalized by the n_positive
        mininal_dinominator = torch.tensor(1.0).to(device) # avoid zero division
        loss = -(labels_attr * torch.log(preds_attr) + (1.0 - labels_attr) * torch.log(1.0 - preds_attr))
        loss = loss.sum(1) / torch.max(mininal_dinominator, n_positive) # (bsz, )
        
        # regularization term
        threshold = torch.tensor(n_positive / n_attributes).to(device)
        loss = loss + self.regularization_loss_fn(
            torch.max(avg_prob_attr, threshold), 
            threshold
        )
        
        if hasattr(self, 'f1_recorder'):
            _, candidates = preds_attr.topk(max(self.topk_list), dim=1, sorted=True, largest=True)
            total_n_positive = labels_attr.sum(1)

            for i, topk in enumerate(self.topk_list):
                this_candidates = candidates[:, :topk]
                this_n_hit = labels_attr.gather(1, this_candidates).sum(1)
                this_n_hit[this_n_hit.eq(0)] = 1e-3
                precision = this_n_hit / topk
                recall = this_n_hit / total_n_positive
                f1 = 2 * precision * recall / (precision + recall)
                
                self.f1_recorder[i].update(f1.sum().item(), f1.size(0), multiply=False)

        if hasattr(self, 'AP_recorder'):
            _, indices = preds_attr.sort(dim=1, descending=True)

            for i in range(labels_attr.shape[0]):
                positive_label = labels_attr[i].nonzero().squeeze(1) # [n_positive_labels]
                all_precisions = 0
                n_success = 0
                for pos, label in enumerate(indices[i]):
                    if label in positive_label:
                        all_precisions += (n_success + 1.0) / (pos + 1)
                        n_success += 1
                        if n_success == len(positive_label):
                            break
                
                average_precision = all_precisions / len(positive_label)
                self.AP_recorder.update(average_precision, 1, multiply=False)

        return loss.sum()
    
    def get_fieldsnames(self, prefix=''):
        return ['%sF1-%02d'%(prefix, item) for item in self.topk_list] \
            + (['%smAP'%prefix] if hasattr(self, 'AP_recorder') else [])

    def get_info(self):
        return self.get_fieldsnames(), \
            [item.avg for item in self.f1_recorder] \
                + ([self.AP_recorder.avg] if hasattr(self, 'AP_recorder') else [])

    def reset_recorder(self):
        self.f1_recorder = [AverageMeter() for _ in range(len(self.topk_list))]
        if self.calculate_mAP:
            self.AP_recorder = AverageMeter()


class NoisyOrMILWithEmbs(NoisyOrMIL):
    def __init__(self, opt, keys, flag, prefix=''):
        if not isinstance(keys, list):
            keys = [keys]
        if len(keys) == 1:
            keys = keys + ['attribute_prediction_prj', 'labels', 'labels_attr']
        else:
            assert len(keys) == 4

        super().__init__(opt, keys=keys)
        self.opt = opt
        self.flag = flag
        self.prefix = prefix

    def get_fieldsnames(self):
        return super().get_fieldsnames(prefix=self.prefix)

    def _step(self, index_indicator, hidden_states, attribute_prediction_prj, labels, labels_attr, *others):
        if isinstance(labels, list):
            labels = labels[-1]

        prj = get_prj_by_flag(self.opt, attribute_prediction_prj, flag=self.flag)
        scores = prj(hidden_states)
        mask = labels.eq(Constants.PAD) # [bsz, seq_len]
        preds_attr, avg_prob_attr = prepare_merged_probs(scores, mask=mask, return_avg_prob=True) # [bsz, n_attributes]
        
        return super()._step(index_indicator, preds_attr, avg_prob_attr, labels_attr)
