"""
Preparation of fingerprint images from different sensors.

The modules of this package need numpy, OpenCV and Pillow only (no torch), so
that the minutiae extraction scripts can use them in the environment of the
minutiae extractor.

sample_table.py describes a training set as a table of Parquet files. It is
imported like the other modules; the package pyarrow is needed only where a
table is read or written.
"""
