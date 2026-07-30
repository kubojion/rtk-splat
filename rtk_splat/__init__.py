"""Georeferenced Gaussian Splatting from stereo and metric pose artifacts.

The mapping core is dataset-agnostic. Dataset adapters may read ROS bags or
external folders, but downstream pose, depth, cloud, and training stages
communicate through the canonical segment artifacts documented in
``docs/architecture/DATA_CONTRACT.md``.
"""

__version__ = "0.1.0"
