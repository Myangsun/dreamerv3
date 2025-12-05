"""Convenience entry point so we can run `python dreamerv3/train.py ...`."""

# This module simply forwards to dreamerv3.main.main().

import os
# Fix cuDNN issues on V100 (Volta architecture) with cuDNN 9.x
# These must be set before importing JAX
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from .main import main


if __name__ == '__main__':  # pragma: no cover
  main()

