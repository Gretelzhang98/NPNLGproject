"""A3DS dataset"""
## the A3DS class is adapt from NPNLG exercise sheet 8.1 by Michael Franke and Polina Tsvilodub
##original dataset at: https://github.com/deepmind/3d-shapes/blob/master/3dshapes_loading_example.ipynb

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import random
from random import shuffle
# from tqdm import tqdm
import pickle
# import 5py

import warnings
warnings.filterwarnings('ignore')

class A3DS(Dataset):
    """
    Dataset class for loading the dataset of images and captions from the 3dshapes dataset.

    Arguments:
    ---------
    num_labels: int
        Number of distinct captions to sample for each image. Relevant for using the dataloader for training models.
    labels_type: str
        "long" or "short". Indicates whether long or short captions should be used.
    run_inference: bool
        Flag indicating whether this dataset will be used for performing inference with a trained image captioner.
    batch_size: int
        Batch size. Has to be 1 in order to save the example image-caption pairs.
    vocab_file: str
        Name of vocab file.
    start_token: str
        Start token.
    end_token: str
        End token.
    unk_token: str
        Token to be used when encoding unknown tokens.
    pad_token: str
        Pad token to be used for padding captions tp max_sequence_length.
    max_sequence_length: int
        Length to which all captions are padded / truncated.
    """
    DATA_SPLITS = set(['train', 'val', 'test'])
    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}
    def __init__(
            self,
            path="A3DS",
            num_labels=1, # number of ground truth labels to retrieve per image
            labels_type="long", # alternative: short
            run_inference=False, # depending on this flag, check presence of model weights
            batch_size=1,
            vocab_file="vocab.pkl",
            start_token="START",  # might be unnecessary since vocab file is fixed anyways
            end_token="END",
            unk_token="UNK",
            pad_token="PAD",
            max_sequence_length=26, # important for padding length
            split="train"
        ):

        # check vocab file exists
        assert os.path.exists(os.path.join(path, vocab_file)), "Make sure the vocab file exists in the directory passed to the dataloader (see README)"

        # check if image file exists
        assert (os.path.exists(os.path.join(path, "sandbox_3Dshapes_1000.pkl")) and os.path.join(path, "sandbox_3Dshapes_resnet50_features_1000.pt")), "Make sure the sandbox dataset exists in the directory passed to the dataloader (see README)"

        if labels_type == "long":
            assert num_labels <= 20, "Maximally 20 distinct image-long caption pairs can be created for one image"
        else:
            assert num_labels <= 27, "Maximally 27 distinct image-short caption pairs can be created for one image"

        self.batch_size = batch_size
        with open(os.path.join(path, vocab_file), "rb") as vf:
            self.vocab = pickle.load(vf)

        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.tokenizer = get_tokenizer("basic_english")

        self.embedded_imgs = torch.load(os.path.join(path, "sandbox_3Dshapes_resnet50_features_1000.pt"))
        with open(os.path.join(path, "sandbox_3Dshapes_1000.pkl"), "rb") as f:
            self.sandbox_file = pickle.load(f)
            self.images = self.sandbox_file["images"]
            self.numeric_labels = self.sandbox_file["labels_numeric"]
            self.labels_long = self.sandbox_file["labels_long"]
            self.labels_short = self.sandbox_file["labels_short"]

        if labels_type == "long":
            labels_ids_flat = [list(np.random.choice(range(len(self.labels_long[0])), num_labels, replace=False)) for i in range(len(self.images))]
            self.labels_flat = [self.labels_long[i][l] for i, sublst in enumerate(labels_ids_flat) for l in sublst]
            self.img_ids_flat = [id for id in range(len(self.images)) for i in range(num_labels)]
        else:
            labels_ids_flat = [list(np.random.choice(range(len(self.labels_short[0])), num_labels, replace=False)) for i in range(len(self.images))]
            self.labels_flat = [self.labels_short[i][l] for i, sublst in enumerate(labels_ids_flat) for l in sublst]
            self.img_ids_flat = [id for id in range(len(self.images)) for i in range(num_labels)]

        # print("len labels ids flat ", len(labels_ids_flat))
        # print("len labels flat ", len(self.labels_flat), self.labels_flat[:5])
        # print("len image ids flat ", len(self.img_ids_flat), self.img_ids_flat[:5])

    def __len__(self):
        """
        Returns length of dataset.
        """
        return len(self.img_ids_flat)

    def __getitem__(self, idx):
        """
        Iterator over the dataset.

        Arguments:
        ---------
        idx: int
            Index for accessing the flat image-caption pairs.

        Returns:
        -------
        target_img: np.ndarray (64,64,3)
            Original image.
        target_features: torch.Tensor(2048,)
            ResNet features of the image.
        target_lbl: str
            String caption.
        numeric_lbl: np.ndarray (6,)
            Original numeric image annotation.
        target_caption: torch.Tensor(batch_size, 25)
            Encoded caption.
        """
        # access raw image corresponding to the index in the entire dataset
        target_img = self.images[self.img_ids_flat[idx]]
        # access caption
        target_lbl = self.labels_flat[idx]
        # access original numeric annotation of the image
        numeric_lbl = self.numeric_labels[self.img_ids_flat[idx]]
        # cast type
        target_img = np.asarray(target_img).astype('uint8')
        # retrieve ResNet features, accessed through original image ID
        target_features = self.embedded_imgs[self.img_ids_flat[idx]]
        # tokenize label
        tokens = self.tokenizer(str(target_lbl).lower().replace("-", " "))
        # Convert caption to tensor of word ids, append start and end tokens.
        target_caption = self.tokenize_caption(tokens)
        # convert to tensor
        target_caption = torch.Tensor(target_caption).long()

        return target_img, target_features, target_lbl, numeric_lbl, target_caption

    def tokenize_caption(self, label):
        """
        Helper for converting list of tokens into list of token IDs.
        Expects tokenized caption as input.

        Arguments:
        --------
        label: list
            Tokenized caption.

        Returns:
        -------
        tokens: list
            List of token IDs, prepended with start, end, padded to max length.
        """
        label = label[:(self.max_sequence_length-2)]
        tokens = [self.vocab["word2idx"][self.start_token]]
        for t in label:
            try:
                tokens.append(self.vocab["word2idx"][t])
            except:
                tokens.append(self.vocab["word2idx"][self.unk_token])
        tokens.append(self.vocab["word2idx"][self.end_token])
        # pad
        while len(tokens) < self.max_sequence_length:
            tokens.append(self.vocab["word2idx"][self.pad_token])

        return tokens

    def get_labels_for_image(self, id, caption_type="long"):
        """
        Helper for getting all annotations for a given image id.

        Arguments:
        ---------
        id: int
            Index of image caption pair containing the image
            for which the full list of captions should be returned.
        caption_type: str
            "long" or "short". Indicates type of captions to provide.

        Returns:
        -------
            List of all captions for given image.
        """
        if caption_type == "long":
            return self.labels_long[self.img_ids_flat[id]]
        else:
            return self.labels_short[self.img_ids_flat[id]]


A3DS_dataset = A3DS()




if __name__ == '__main__':
    A3DS_dataset = A3DS()
    print(A3DS_dataset.__len__())
    itemID = 0
    image, target_features, caption_text, numeric_lbl, caption_indx = A3DS_dataset.__getitem__(itemID)


    # picture
    print(image)

    #numeric annotation
    print(numeric_lbl)

    # plot image
    plt.imshow(image)
    plt.show()

    # ground-truth caption
    print(caption_text)

    # Retrieve all short-captions for the image ID:
    all_short_caps = A3DS_dataset.get_labels_for_image(itemID, caption_type='short')
    for c in all_short_caps:
        print(c)

    # Retrieve all long-captions for the image ID:

    all_long_caps = A3DS_dataset.get_labels_for_image(itemID, caption_type='long')
    for c in all_long_caps:
        print(c)

    vocab = A3DS_dataset.vocab["word2idx"].keys()
    print("VOCAB: ", vocab)

    vocab_dict = A3DS_dataset.vocab["word2idx"]
    print(vocab_dict)

    vocab_size = len(vocab)
    print("VOCAB SIZE: ", vocab_size)