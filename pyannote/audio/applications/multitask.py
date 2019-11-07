#!/usr/bin/env python
# encoding: utf-8

"""
Speaker maturity detection

Usage:
  pyannote-multitask train [options] <experiment_dir> <database.task.protocol>
  pyannote-multitask validate [options] [--every=<epoch> --chronological] <train_dir> <database.task.protocol>
  pyannote-multitask apply [options] [--step=<step>] <validate_dir> <database.task.protocol>
  pyannote-multitask -h | --help
  pyannote-multitask --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "AMI.SpeakerDiarization.MixHeadset")
  --database=<database.yml>  Path to pyannote.database configuration file.
  --subset=<subset>          Set subset (train|developement|test).
                             Defaults to "train" in "train" mode. Defaults to
                             "development" in "validate" mode. Defaults to
                             "test" in "apply" mode.
  --gpu                      Run on GPUs. Defaults to using CPUs.
  --batch=<size>             Set batch size. Has no effect in "train" mode.
                             [default: 32]
  --from=<epoch>             Start {train|validat}ing at epoch <epoch>. Has no
                             effect in "apply" mode. [default: 0]
  --to=<epochs>              End {train|validat}ing at epoch <epoch>.
                             Defaults to keep going forever.

"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.

"validation" mode:
  --every=<epoch>            Validate model every <epoch> epochs [default: 1].
  --chronological            Force validation in chronological order.
  --parallel=<n_jobs>        Process <n_jobs> files in parallel. Defaults to
                             using all CPUs.
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).

"apply" mode:
  <validate_dir>             Path to the directory containing validation
                             results (i.e. the output of "validate" mode).
  --step=<step>              Sliding window step, in seconds.
                             Defaults to 25% of window duration.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process,
    the neural network architecture, and the task addressed.

    ................... <experiment_dir>/config.yml ...................
    # train the network for speech activity detection
    # see pyannote.audio.labeling.tasks for more details
    task:
       name: Multitask
       params:
          duration: 3.2     # sub-sequence duration
          per_epoch: 1      # 1 day of audio per epoch
          batch_size: 32    # number of sub-sequences per batch

    # use precomputed features (see feature extraction tutorial)
    feature_extraction:
       name: Precomputed
       params:
          root_dir: tutorials/feature-extraction

    # use the StackedRNN architecture.
    # see pyannote.audio.labeling.models for more details
    architecture:
       name: StackedRNN
       params:
         rnn: LSTM
         recurrent: [16, 16]
         bidirectional: True

    # use cyclic learning rate scheduler
    scheduler:
       name: CyclicScheduler
       params:
           learning_rate: auto
    ...................................................................

"train" mode:
    This will create the following directory that contains the pre-trained
    neural network weights after each epoch:

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "validate" mode.

    A bunch of values (loss, learning rate, ...) are sent to and can be
    visualized with tensorboard with the following command:

        $ tensorboard --logdir=<experiment_dir>

"validate" mode:
    Use the "validate" mode to run validation in parallel to training.
    "validate" mode will watch the <train_dir> directory, and run validation
    experiments every time a new epoch has ended. This will create the
    following directory that contains validation results:

        <train_dir>/validate/<database.task.protocol>.<subset>

    You can run multiple "validate" in parallel (e.g. for every subset,
    protocol, task, or database).

    In practice, for each epoch, "validate" mode will look for the detection
    threshold that minimizes the detection error rate.

"apply" mode:
    Use the "apply" mode to extract speaker maturity detection raw scores and
    results. This will create the following directory that contains speech
    activity detection results:

        <validate_dir>/apply/<epoch>
"""

from functools import partial
from pathlib import Path
import torch
import numpy as np
import scipy.optimize
from docopt import docopt
import multiprocessing as mp
from .base_labeling import BaseLabeling
from pyannote.database import get_annotated
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.pipeline import SpeechActivityDetection \
                             as SpeechActivityDetectionPipeline
import pdb

class Multitask(BaseLabeling):
    pass


def main():
    arguments = docopt(__doc__, version='Multitask classification')

    db_yml = arguments['--database']
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    gpu = arguments['--gpu']
    device = torch.device('cuda') if gpu else torch.device('cpu')

    # HACK to "book" GPU as soon as possible
    _ = torch.Tensor([0]).to(device)

    if arguments['train']:
        experiment_dir = Path(arguments['<experiment_dir>'])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        if subset is None:
            subset = 'train'

        # start training at this epoch (defaults to 0)
        restart = int(arguments['--from'])

        # stop training at this epoch (defaults to never stop)
        epochs = arguments['--to']
        if epochs is None:
            epochs = np.inf
        else:
            epochs = int(epochs)

        application = Multitask(experiment_dir, db_yml=db_yml,
                                              training=True)
        application.device = device
        application.train(protocol_name, subset=subset,
                          restart=restart, epochs=epochs)