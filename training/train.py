"""
Trainer factory: dispatches to the segmentation trainer.
"""


def get_trainer(model, config, metrics_tracker):
    """
    Returns the appropriate trainer for the given model and config.
    All models use SegmentationTrainer for QPI segmentation.
    """
    from training.trainer_seg import SegmentationTrainer
    return SegmentationTrainer(model, config, metrics_tracker)