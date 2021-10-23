import numpy as np


def voc_bbox_overlaps(bboxes1: np.ndarray,
                      bboxes2: np.ndarray,
                      mode: str = 'iou',
                      eps: float = 1e-6) -> np.ndarray:
    """Calculate the ious between each bbox of bboxes1 and bboxes2. The
    calculation follows the official evaluation code of Pascal VOC dataset.

    Args:
        bboxes1 (np.ndarray): With shape (n, 4).
        bboxes2 (np.ndarray): With shape (k, 4).
        mode (str): IoU (intersection over union) or iof (intersection
            over foreground).
        eps (float): Constant variable to avoid division by zero.
    Returns:
        ious (np.ndarray): With Shape (n, k).
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1.0) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1.0)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1.0) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1.0)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1.0, 0) * np.maximum(
            y_end - y_start + 1.0, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
