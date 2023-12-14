Two deep learning architectures for classifying and
segmenting the flower dataset in the Visual Geometry Group
database are proposed. Custom model architectures have been
developed rather than relying on large pre-trained models.
The architecture design is inspired by the ideas from the
Resnet [1], Unet [2], and Fully Convolutional Networks(FCN)
[3] architecture. For the classification task, we have used a
convolutional skip connection to propagate the output from the
previous layer, an idea inspired by the Residual net architecture.
For the segmentation task, instead of plain down sampling
and up sampling network, we have concatenated the output of
downsampling to the output of upsampling to preserve the spatial
information.
