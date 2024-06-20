# r3d_v0_embeddings
This repository contains vector embeddings for the EchoNet dataset and the trained model that extracted them. It is intended as a resource for future AI researchers in Cardiology.

# The Embeddings
In the base directory there is one file containing the best 400-dimensional embeddings learned in the paper
* `echonet_embeddings.txt.gz`: 400-dimensional embeddings of the 10,030 echocardiogram videos from the EchoNet dataset, learned from the R3D transformer model we developed in our paper. Each embedding vector is identifiable by the video ID of the echocardiogram it represents. Those video IDs and the entire EchoNet dataset can be found at this link: https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a

# The Model
How to generate your own echo embeddings:
* Navigate to your workspace or desktop
* `git clone git@github.com:Team-Echo-MIT/r3d-v0-embeddings.git` - clone this repository into your workspace
* Navigate to the cloned repository in your workspace
* Put echocardiograms to embed as .avi files in the `embedder/echos` subdirectory
* Run the extraction script in the terminal: `python generate_echo_embeddings.py <your-path>/embedder/echos <your-path>/embedder/r3d_binary_111723.pt <your-path>/embedder/tensor_board` (replace `<your-path>` with the path leading to the embedder directory in your workspace)
* Embeddings should be written to a txt file in the `embedder/embeddings` subdirectory

Repository Name Etymology
* `R3D` - name of the transformer model used to learn the embeddings
* `V0` - version of the model (this is the first version of our model)
* `embeddings` - our focus is on the embeddings that the model learned for each echocardiogram



Contributors:
> Daniel Chung `djaechung` <br /> Mindy Somin Lee `mindyslee` <br /> Vasu Kaker `VasuKaker` <br /> Yongyi Zhao
