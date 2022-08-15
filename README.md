# merfish_cambridge_preprocessing

This repository contains script for splitting the original merged merFISH data produced by the Cambridge lab.

# Scripts functionalities

1. Execute cambridge_splitter (default clahe_adjust = False) to split original merged data into specific FOVs and slices.

2. Execute cambridge_splitter (set clahe_adjust = True) to split original merged data into specific FOVs and slices, with CLAHE applied

3. Execute subdivide.py (dependent on utils.py and sparse.py) from AnglerFISH simulator to chop specific FOVs and slices images into small pieces to feed through the AnglerFISH pre-trained model.
