import numpy as np

def compute_mIoU(preds, labels, num_classes):
    ious = []
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return np.nanmean(ious), ious
