import numpy as np
import torch


def label_wrapper(labels, class_ids):
    """Map input labels into range of 0 to numbers of classes-1.

    It is usually used in the meta testing phase.

    Args:
        labels (Tensor | np.ndarray | list): The labels to be wrapped.
        class_ids (list[int]): All class ids of labels.

    Returns:
        (Tensor | np.ndarray | list): Same type as the input labels.
    """
    class_id_map = {class_id: i for i, class_id in enumerate(class_ids)}
    if isinstance(labels, torch.Tensor):
        wrapped_labels = torch.tensor(
            [class_id_map[label.item()] for label in labels])
        wrapped_labels = wrapped_labels.type_as(labels).to(labels.device)
    elif isinstance(labels, np.ndarray):
        wrapped_labels = np.array([class_id_map[label] for label in labels])
        wrapped_labels = wrapped_labels.astype(labels.dtype)
    elif isinstance(labels, (tuple, list)):
        wrapped_labels = [class_id_map[label] for label in labels]
    else:
        raise TypeError('only support torch.Tensor, np.ndarray and list')
    return wrapped_labels
