import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_MULTI_CLASS_CLASSIFICATION
from ..gradient_reversal import GradientReversal
from pyannote.audio.models.models import RNN

class MultitaskGenerator(LabelingTaskGenerator):
    """Batch generator for training speech activity detection

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    frame_info : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
    frame_crop : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models
        that include the feature extraction step (e.g. SincNet) and
        therefore use a different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def postprocess_y(self, Y):
        """Generate labels for speech activity detection

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by
            `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """


        return Y

    @property
    def specifications(self):
        specs = {
            'task': TASK_MULTI_CLASS_CLASSIFICATION,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': ['child', 'non_child']},
        }
        for key, classes in self.file_labels_.items():
            specs[key] = {'classes': classes}

        return specs

class Multitask(LabelingTask):

    SPEECH_PT = '{log_dir}/weights/{epoch:04d}.domain.pt'
    ADULT_PT = '{log_dir}/weights/{epoch:04d}.adult.pt'
    GENDER_PT = '{log_dir}/weights/{epoch:04d}.gender.pt'
    CHILD_PT = '{log_dir}/weights/{epoch:04d}.child.pt'

    def __init__(self, label_spec, subtasks=None, **kwargs):
        super().__init__(**kwargs)
        self.label_spec = label_spec
        self.subtasks = subtasks

        for _, subtask in self.subtasks.items():
            subtask['loss_func'] = getattr(F, subtask['loss_func'])
            subtask['activation'] = getattr(nn, subtask['activation'])()

    def parameters(self, model, specifications, device):
        parameters = []
        for key,subtask in self.subtasks.items():

            subtask_classifier_rnn = RNN(
                n_features=model.intermediate_dimension(subtask['attachment']),
                **subtask['rnn'])

            subtask_classifier_linear = nn.Linear(
                subtask_classifier_rnn.dimension,
                2,
                bias=True).to(device)

            subtask['classifier'] = nn.Sequential(subtask_classifier_rnn,
                                                                 subtask_classifier_linear).to(device)

            parameters += list(subtask['classifier'].parameters())

        return parameters

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """

        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device='cpu')

        attachments = [subtask['attachment'] for _, subtask in self.subtasks.items()]

        fX, intermediates = self.model_(X, return_intermediate=attachments)

        for _, subtask in self.subtasks.items():
            labels_indexes = [self.batch_generator_.segment_labels_.index(label) for label in subtask['labels']]
            subtask_target = np.sum(batch['y'][:, :, labels_indexes], axis=2) > 0
            subtask_target = torch.tensor(subtask_target, dtype=torch.int64, device=torch.device('cpu')).view((-1,))
            subtask_scores = F.log_softmax(subtask['classifier'](intermediates[subtask['attachment']]).contiguous().view((-1, 2)))
            subtask['loss'] = subtask['loss_func'](subtask_scores,subtask_target)

        losses = {'loss_' + key : subtask['loss'] for key,subtask in self.subtasks.items()}
        losses['loss'] = sum(losses[loss] for loss in losses)
        # losses['loss_task'] = loss

        return losses



    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            frame_info=None, frame_crop=None):

        return MultitaskGenerator(
            feature_extraction,
            protocol, subset=subset,
            frame_info=frame_info,
            frame_crop=frame_crop,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            parallel=self.parallel)

