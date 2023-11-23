# r3d_v0_embeddings
This repository contains code accompanying publication submission of the paper: 
> M. Lee, D. Chung, V. Kaker, Y. Zhao, S. Perera, P. Sasankan, I. Riaz, G. Tang, K. Jacques, P. Kuo, B. Kazzi, L. Celi [Deep Learning Prediction of Ejection Fraction from Echocardiograms: Vector Embeddings from a Best Practice R3D Transformer] In consideration for the IEEE Instrumentation and Measurement for a Sustainable Future (I2MTC) conference, 2024.

In the base directory there is one file containing the best 400-dimensional embeddings learned in the paper
* `echonet_embeddings.txt.gz`: 400-dimensional embeddings of the 10,030 echocardiogram videos from the EchoNet dataset, learned from the R3D transformer model we developed in our paper. Each embedding vector is identifiable by the video ID of the echocardiogram it represents. Those video IDs and the entire EchoNet dataset can be found at this link: https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a

File Etymology
* `R3D` - name of the transformer model used to learn the embeddings
* `V0` - version of the model (this is the first version of our model)
* `embeddings` - our focus is on the embeddings that the model learned for each echocardiogram
