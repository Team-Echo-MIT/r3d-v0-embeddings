import argparse
import torch
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class r3dmodel(nn.Module):
  def __init__(self, model1, regression = False):
    super(r3dmodel, self).__init__()
    self.regression = regression
    self.preloaded_model = model1
    self.new_layer1 = nn.Linear(400,1)
    if self.regression == False:
        self.new_layer2 = nn.Sigmoid()
  def forward(self, x):
    x = self.preloaded_model(x)
    x = self.new_layer1(x)
    if self.regression == False:
        x = self.new_layer2(x)
    return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class DeepFeatures(torch.nn.Module):
    '''
    This class extracts, reads, and writes data embeddings using a pretrained deep neural network. Meant to work with
    Tensorboard's Embedding Viewer (https://www.tensosrflow.org/tensorboard/tensorboard_projector_plugin).
    When using with a 3 channel image input and a pretrained model from torchvision.models please use the
    following pre-processing pipeline:

    transforms.Compose([transforms.Resize(imsize),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) ## As per torchvision docs

    Args:
        model (nn.Module): A Pytorch model that returns an (B,1) embedding for a length B batched input
        imgs_folder (str): The folder path where the input data elements should be written to
        embs_folder (str): The folder path where the output embeddings should be written to
        tensorboard_folder (str): The folder path where the resulting Tensorboard log should be written to
        experiment_name (str): The name of the experiment to use as the log name
    '''

    def __init__(self, model,
                 imgs_folder,
                 embs_folder,
                 tensorboard_folder,
                 experiment_name=None):

        super(DeepFeatures, self).__init__()

        self.model = model
        self.model.eval()

        self.imgs_folder = imgs_folder
        self.embs_folder = embs_folder
        self.tensorboard_folder = tensorboard_folder

        self.name = experiment_name

        self.writer = None

    def generate_embeddings(self, x):
        '''
        Generate embeddings for an input batched tensor
        Args:
            x (torch.Tensor) : A batched pytorch tensor
        Returns:
            (torch.Tensor): The output of self.model against x
        '''
        return(self.model(x))

    def write_embeddings(self, x, input_id, output_dir, outsize=(28,28)):
        '''
        Generate embeddings for an input batched tensor and write inputs and
        embeddings to self.imgs_folder and self.embs_folder respectively.

        Inputs and outputs will be stored in .npy format with randomly generated
        matching filenames for retrieval

        Args:
            x (torch.Tensor) : An input batched tensor that can be consumed by self.model
            outsize (tuple(int, int)) : A tuple indicating the size that input data arrays should be
            written out to

        Returns:
            (bool) : True if writing was succesful
        '''

        #assert len(os.listdir(self.imgs_folder))==0, "Images folder must be empty"
        #assert len(os.listdir(self.embs_folder))==0, "Embeddings folder must be empty"

        # Generate embeddings
        embs = self.generate_embeddings(x)

        # Detach from graph
        embs = embs.detach().cpu().numpy().flatten().tolist()

        # Write embeddings into embedddings list, tagged with unique video id
        embs_txt_path = f'{output_dir}/embeddings_revised.txt'

        # file = open(embs_txt_path,"w")
        with open(embs_txt_path,"a") as outfile:
          outfile.write(f"{input_id} ")
          for embedding in embs:
            outfile.write(f"{str(embedding)} ")
          outfile.write("\n")
          outfile.close()
        return(True)


    def _create_writer(self, name):
        '''
        Create a TensorboardX writer object given an experiment name and assigns it to self.writer
        Args:
            name (str): Optional, an experiment name for the writer, defaults to self.name
        Returns:
            (bool): True if writer was created succesfully
        '''

        if self.name is None:
            name = 'Experiment_' + str(np.random.random())
        else:
            name = self.name

        dir_name = os.path.join(self.tensorboard_folder, name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        else:
            print("Warning: logfile already exists")
            print("logging directory: " + str(dir_name))

        logdir = dir_name
        self.writer = SummaryWriter(logdir=logdir)
        return(True)

    def create_tensorboard_log(self):
        '''
        Write all images and embeddings from imgs_folder and embs_folder into a tensorboard log
        '''

        if self.writer is None:
            self._create_writer(self.name)

        ## Read in
        all_embeddings = [np.load(os.path.join(self.embs_folder, p)) for p in os.listdir(self.embs_folder) if p.endswith('.npy')]
        all_images = [np.load(os.path.join(self.imgs_folder, p)) for p in os.listdir(self.imgs_folder) if p.endswith('.npy')]
        all_images = [np.moveaxis(a, 2, 0) for a in all_images] # (HWC) -> (CHW)

        ## Stack into tensors
        all_embeddings = torch.Tensor(all_embeddings)
        all_images = torch.Tensor(all_images)

        print(all_embeddings.shape)
        print(all_images.shape)

        self.writer.add_embedding(all_embeddings, label_img = all_images)

def tensor2np(tensor, resize_to=None):
    '''
    Convert an image tensor to a numpy image array and resize
    Args:
        tensor (torch.Tensor): The input tensor that should be converted
        resize_to (tuple(int, int)): The desired output size of the array
    Returns:
        (np.ndarray): The input tensor converted to a channel last resized array
    '''
    out_array = tensor.detach().cpu().numpy()
    out_array = np.moveaxis(out_array, 0, 2) # (CHW) -> (HWC)
    if resize_to is not None:
        out_array = cv2.resize(out_array, dsize=resize_to, interpolation=cv2.INTER_CUBIC)
    return(out_array)


def avi_to_tensor(video_file, max_frames=None):
    """Convert an avi file to a tensor object.
    Args:
        video_file (str): path to the avi file to convert
        max_frames (int): cap for number of frames to convert
    Returns:
        PyTorch tensor representation of the avi video
    """
    # initialize a list to store video frames
    frames = []
    # open the video file
    cap = cv2.VideoCapture(video_file)
    # read frames from the video file
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # transpose the frame shape from (height, width, channel) to 
        # (channel, height, width)
        frame_t = np.transpose(frame_rgb, (2, 0, 1))
        frames.append(frame_t)
        # stop reading frames if maximum number of frames is reached
        if max_frames is not None and len(frames) >= max_frames:
            break
    # release the video file
    cap.release()
    # stack the frames to create a 4D numpy array
    video_array = np.stack(frames, axis=0)
    # convert the numpy array to a PyTorch tensor
    video_tensor = torch.from_numpy(video_array).float()
    return video_tensor


def create_embeddings(echo_dir, model_path, output_dir, tensorboard_dir):
    # import the R3D model
    weights1 = R3D_18_Weights.DEFAULT
    model1 = r3d_18(weights=weights1)
    model1.eval()
    model_r3d = r3dmodel(model1)
    model = model_r3d
    model_parameters = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_parameters)
    # modify it with identity layers
    model._modules['new_layer1'] = Identity()
    model._modules['new_layer2'] = Identity()

    # create the deep features extractor
    emb_fetcher = DeepFeatures(model, echo_dir, output_dir, tensorboard_dir)

    filenames = os.listdir(echo_dir)
    # for every file in echo_dir:
    i = 0
    for filename in filenames:
        i += 1
        if i % 10 == 0:
            print(f"Captured Embeddings for {i} of {len(filenames)} Videos")
        if filename.endswith('avi'):
            # convert AVI to tensor
            file_path = (f"{echo_dir}/{filename}")
            echocardiogram_tensor = avi_to_tensor(file_path)
            echocardiogram_tensor = echocardiogram_tensor.unsqueeze(2).permute(2,1,0,3,4)
            # write the embeddings
            emb_fetcher.write_embeddings(x=echocardiogram_tensor, input_id=filename, output_dir=output_dir)
    pass


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for echocardiograms.')
    parser.add_argument('echo_dir', type=str, help='Directory containing echocardiograms to embed')
    parser.add_argument('model_path', type=str, help='Path to the embedding model')
    parser.add_argument('output_dir', type=str, help='Directory where the resulting Tensorboard log should be written to')
    parser.add_argument('tensorboard_dir', type=str, help='Directory where the txt file of echocardiogram embeddings will live')
    
    args = parser.parse_args()
    
    # Call the create_embeddings function with the input data
    create_embeddings(args.echo_dir, args.model_path, args.output_dir, args.tensorboard_dir)


if __name__ == '__main__':
    main()