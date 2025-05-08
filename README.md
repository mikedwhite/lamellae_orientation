# lamellae_orientation

Python script for automated measurement of lamellae orientation, relative to the bottom edge of the image field.
The images are inverted, white balanced, and binarised before watershed is used to extract individual frames.
The eigenvectors of the covariance matrix for each grain then gives the lamella direction.

This code requires the microstructural-fingerprinting-tools package, which can be found [here](https://github.com/mikedwhite/microstructural-fingerprinting-tools).
