import math
import os
import time
from ast import literal_eval
from datetime import datetime

import clip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as Datasets
import torchvision.models as models
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange, tqdm

from vgg19 import VGG19

torch.manual_seed(0)
clip_model = None
# ---------------------------------------------------------------
date_time_obj = datetime.now()
timestamp_str = date_time_obj.strftime("%Y-%m-%d_%H.%M.%S")
print('Current Timestamp : ', timestamp_str)
start_time_total = time.perf_counter()
# ---------------------------------------------------------------
# Parameter you can change


batch_size = 128
num_epoch = 25
save_model_every = 5
# Available datasets: celeba_small, celeba, celeba_resize
dataset = "celeba_resize"  # < --------------change this!
print("dataset = ", dataset)
latent_dim = 128  # < --------------change this!

# ---------------------------------------------------------------
# !! no "pt"
# set to None if you do not want to load a checkpoint
#
load_checkpoint = None  # "CNN_VAE_celeba_2022-08-04_23.22.32"
run_train = True # < --------------change this!

# Checkpoint os pretrained models
# "CNN_VAE_celeba_small_2022-08-04_21.40.10" (small, for testing)
# "CNN_VAE_celeba_2022-08-04_18.05.43" (large model trained for 20 epochs)
# "celebA_64_8_epoch_latent_dim_128"
# "CNN_VAE_celeba_2022-08-04_23.22.32" (add condition to layer 4, trained for 10 epochs, good result, latent = 128)
# RES_VAE2, run2
# 10 epochs version is in the backup
# CNN_VAE_celeba_2022-08-05_01.49.10 #(VAE3) (add condition to layer 5)
# CNN_VAE_celeba_2022-08-05_03.20.52  #python run2.py (add condition to layer 4)  VAE2 , latent dim = 128
# CNN_VAE_celeba_2022-08-05_11.31.54 # run2.py (add condition to layer 4)  VAE2, latent dim = 512 (quality not good,
# 10 epoch)


# continue training  "CNN_VAE_celeba_2022-08-04_23.22.32" for 10 epochs, ...
# CNN_VAE_celeba_2022-08-05_14.32.29  # run4 (condition only in latent) #not very good
# replicate the best model again, find epoch that is best
# CNN_VAE_celeba_2022-08-05_17.20.03 # run5, latent 128 # not good recon and generate result
# CNN_VAE_celeba_2022-08-05_18.07.11 # run6, latent 512 # better recon, text to image is better, blur but generate
# same faces
# continue 10 epoch each
# ---------------------------------------------------------------
# logging
if run_train and load_checkpoint:
    print("Train with pretrained model...")
elif run_train and load_checkpoint is None:
    print("Train from scratch...")
elif load_checkpoint is not None and not run_train:
    print("Only load pretrained model, do not train...")
elif load_checkpoint is None and not run_train:
    # print("Set run_train to True or give a checkpoint")
    raise SystemExit("!Set run_train to True or give a checkpoint...")

# ---------------------------------------------------------------
# Parameters you may NOT want to change
condition_dim = 512
image_size = 64
lr = 1e-4
start_epoch = 0
dataset_root = "./input/"
save_dir = os.getcwd()
beta = 0.1

# ---------------------------------------------------------------
use_cuda = torch.cuda.is_available()
GPU_indx = 0
device = torch.device(GPU_indx if use_cuda else "cpu")

# ---------------------------------------------------------------
if load_checkpoint:
    model_name = load_checkpoint
else:
    model_name = f"CNN_VAE_{dataset}_{timestamp_str}"  # "STL10_8" #"STL10_8" #STL10_8_64.pt

if model_name == "CNN_VAE_celeba_2022-08-05_01.49.10":
    from RES_VAE_conditioned_encoder_layer5 import VAE
    # latent_dim = 512
elif model_name in ["CNN_VAE_celeba_2022-08-05_03.20.52", "CNN_VAE_celeba_2022-08-05_11.31.54"]:
    from RES_VAE_conditioned_encoder_layer4 import VAE
    # latent_dim = 512

elif model_name in ["CNN_VAE_celeba_2022-08-04_23.22.32"]:  # best model
    from RES_VAE_conditioned_encoder_layer4 import VAE
    # latent_dim = 128

elif model_name in ["CNN_VAE_celeba_2022-08-05_14.32.29"]:
    from RES_VAE_conditioned_only_latent import VAE
else:
    from RES_VAE_conditioned import VAE


# ---------------------------------------------------------------
class CelebA_CLIP(Datasets.ImageFolder):
    def __init__(
            self,
            root,
            transform,
            image_folder,
            clip_embeddings_csv
    ):
        super(CelebA_CLIP, self).__init__(
            root=root,
            transform=transform
        )

        self.clip_embeddings = clip_embeddings_csv
        self.samples = self.make_dataset_(root, None, None, None)
        self.root = os.path.join(root, image_folder)

    def __len__(self) -> int:
        return len(self.samples)

    def make_dataset_(self, root, class_to_idx, extensions, is_valid_file):
        df = pd.read_csv(self.clip_embeddings, index_col=0,
                         converters={'embeddings': literal_eval})
        im_names = df['image_id'].values
        # img_embed = zip(df['image_id'].values, df["embeddings"].values)
        # img_embed = tuple(zip(range(len(im_names)), im_names, df["embeddings"].values))
        targets = df["embeddings"].values  # <class 'numpy.ndarray'> #(batch,)
        img_embed = tuple(zip(im_names, targets))

        return list(img_embed)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # print(len(self.samples))
        # print("index,", index)
        path, target = self.samples[index]
        # print("path", path)
        path = os.path.join(self.root, path)
        # print("path", path)

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(target)

        return sample, target


# ---------------------------------------------------------------
def get_data_STL10(transform, batch_size, download=True, root="./input"):
    print("Loading trainset...")
    trainset = Datasets.STL10(root=root, split='unlabeled', transform=transform, download=download)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    print("Loading testset...")
    testset = Datasets.STL10(root=root, split='test', download=download, transform=transform)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("Done!")

    return trainloader, testloader


def get_data_celebA(transform, batch_size):
    # data_root = "../../datasets/celeba_small/celeba/"
    data_root = "../datasets/celeba/"
    training_data = CelebA_CLIP(root=data_root,
                                transform=transform,
                                image_folder="img_align_celeba",
                                clip_embeddings_csv="../conditional_VAE/src/embeddings.csv")
    print("dataset size", len(training_data))  # 202599
    test_size = 16
    train_size = len(training_data) - test_size
    trainset, testset = torch.utils.data.random_split(training_data, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    print("Done load dataset")
    return trainloader, testloader, train_size


def get_data_celebA_resized(transform, batch_size):
    # data_root = "../../datasets/celeba_small/celeba/"
    data_root = "../datasets/resized_celebA3/"
    training_data = CelebA_CLIP(root=data_root,
                                transform=transform,
                                image_folder="celebA",
                                clip_embeddings_csv="../conditional_VAE/src/embeddings.csv")
    print("dataset size", len(training_data))  # 202599
    test_size = 16
    train_size = len(training_data) - test_size
    trainset, testset = torch.utils.data.random_split(training_data, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    print("Done load dataset")
    return trainloader, testloader, train_size


def get_data_celebA_small(transform, batch_size):
    data_root = "../datasets/celeba_small/celeba/"
    training_data = CelebA_CLIP(root=data_root,
                                transform=transform,
                                image_folder="img_align_celeba",
                                clip_embeddings_csv="./embeddings_128.csv")

    print("dataset size", len(training_data))  # 128
    test_size = 16
    train_size = len(training_data) - test_size
    trainset, testset = torch.utils.data.random_split(training_data, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    print("Done load dataset")
    return trainloader, testloader, train_size


# ---------------------------------------------------------------
if dataset == "celeba":
    transform = T.Compose([T.CenterCrop(178),T.Resize((image_size,image_size)), T.ToTensor()])
    #transform = T.Compose([T.Resize((image_size,image_size)), T.ToTensor()])
    #transform = T.Compose([T.Resize(image_size), T.ToTensor()])
    #transform = T.Compose([T.ToTensor()])
    trainloader, testloader, train_size = get_data_celebA(transform, batch_size)


if dataset == "celeba_resize":
    #transform = T.Compose([T.CenterCrop(178),T.Resize((image_size,image_size)), T.ToTensor()])
    #transform = T.Compose([T.Resize((image_size,image_size)), T.ToTensor()])
    #transform = T.Compose([T.Resize(image_size), T.ToTensor()])
    transform = T.Compose([T.ToTensor()])
    trainloader, testloader, train_size = get_data_celebA(transform, batch_size)

elif dataset == "celeba_small":
    transform = T.Compose([T.CenterCrop(178), T.Resize((image_size, image_size)), T.ToTensor()])
    trainloader, testloader, train_size = get_data_celebA_small(transform, batch_size)
else:
    transform = T.Compose([T.Resize(image_size), T.ToTensor()])
    trainloader, testloader = get_data_STL10(transform, batch_size, download=True, root=dataset_root)

# ---------------------------------------------------------------
# get a test image batch from the testloader to visualise the reconstruction quality
dataiter = iter(testloader)
test_images, test_labels = dataiter.next()
print("load test batch")
print("image input shape", test_images.shape)
print("condition shape", test_labels.shape)  # torch.Size([16, 1, 512])


# ---------------------------------------------------------------
# OLD way of getting features and calculating loss - Not used

# create an empty layer that will simply record the feature map passed to it.
class GetFeatures(nn.Module):
    def __init__(self):
        super(GetFeatures, self).__init__()
        self.features = None

    def forward(self, x):
        self.features = x
        return x


# download the pre-trained weights of the VGG-19 and append them to an array of layers .
# we insert a GetFeatures layer after a relu layer.
# layers_deep controls how deep we go into the network
def get_feature_extractor(layers_deep=7):
    C_net = models.vgg19(pretrained=True).to(device)
    C_net = C_net.eval()

    layers = []
    for i in range(layers_deep):
        layers.append(C_net.features[i])
        if isinstance(C_net.features[i], nn.ReLU):
            layers.append(GetFeatures())
    return nn.Sequential(*layers)


# this function calculates the L2 loss (MSE) on the feature maps copied by the layers_deep
# between the reconstructed image and the origional
def feature_loss(img, recon_data, feature_extractor):
    img_cat = torch.cat((img, torch.sigmoid(recon_data)), 0)
    out = feature_extractor(img_cat)
    loss = 0
    for i in range(len(feature_extractor)):
        if isinstance(feature_extractor[i], GetFeatures):
            loss += (feature_extractor[i].features[:(img.shape[0])] - feature_extractor[i].features[
                                                                      (img.shape[0]):]).pow(2).mean()
    return loss / (i + 1)


# Linear scaling the learning rate down
def lr_Linear(epoch_max, epoch, lr):
    lr_adj = ((epoch_max - epoch) / epoch_max) * lr
    set_lr(lr=lr_adj)


def set_lr(lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(recon, x)
    KL_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    loss = recon_loss + 0.01 * KL_loss
    return loss


# ---------------------------------------------------------------

# Create the feature loss module

# load the state dict for vgg19
state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
# manually create the feature extractor from vgg19
feature_extractor = VGG19(channel_in=3)

# loop through the loaded state dict and our vgg19 features net,
# loop will stop when net.parameters() runs out - so we never get to the "classifier" part of vgg
for ((name, source_param), target_param) in zip(state_dict.items(), feature_extractor.parameters()):
    target_param.data = source_param.data
    target_param.requires_grad = False

feature_extractor = feature_extractor.to(device)

# ---------------------------------------------------------------

# Create the save directory if it does note exist
if not os.path.isdir(save_dir + "/Models"):
    os.makedirs(save_dir + "/Models")
if not os.path.isdir(save_dir + "/Results"):
    os.makedirs(save_dir + "/Results")

result_folder = os.path.join(save_dir, "Results", f"result_{model_name}")
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

# ---------------------------------------------------------------
# Load / Initialize the model
model_save_path = os.path.join(save_dir, "Models", model_name + ".pt")

if load_checkpoint:

    if model_name == "CNN_VAE_celeba_2022-08-04_18.05.43":
        batch_size = 128
        condition_dim = 512
        latent_dim = 512
        checkpoint = torch.load(save_dir + "/Models/" + model_name + ".pt", map_location="cpu")
        print("Checkpoint loaded")
        vae_net = VAE(channel_in=3 + condition_dim, ch=64, z=latent_dim, condition_dim=condition_dim).to(device)
        optimizer = optim.Adam(vae_net.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        vae_net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint["epoch"]
        loss_log = checkpoint["loss_log"]

    else:
        vae_net = torch.load(model_save_path)
    # print("....mu", vae_net.mu.shape)
    # print("....logvar", vae_net.log_var.shape)

elif run_train:
    # If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(model_save_path):
        # raise ValueError("Warning Checkpoint exists")
        print("Warning Checkpoint exists")

    # Create VAE network
    # z = latent dim, ch = out channel
    print("Initialize VAE net ...")
    vae_net = VAE(channel_in=3 + condition_dim, ch=64, z=latent_dim, condition_dim=condition_dim).to(device)

# setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=lr, betas=(0.5, 0.999))
# Loss function
BCE_Loss = nn.BCEWithLogitsLoss()


# ---------------------------------------------------------------


def convert_batch_to_image_grid(image_batch, dim=64):
    print("image_batch", image_batch.shape)
    # torch.Size([16, 3, 64, 64])
    reshaped = (image_batch.reshape(4, 8, dim, dim, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(4 * dim, 8 * dim, 3))
    return reshaped


import torchvision


def save_image_grid(img_tensor, save_path):
    if torch.is_tensor(img_tensor):
        img_tensor = img_tensor.cpu()

    grid_img = torchvision.utils.make_grid(img_tensor, scale_each=True)
    plt.figure(figsize=(20, 20))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    plt.savefig(os.path.join(save_path), dpi=100, bbox_inches='tight')


def save_each_image(img_tensor):
    img_tensor = img_tensor.detach()
    for i in range(img_tensor.shape[0]):
        img = img_tensor[i].permute(1, 2, 0)
        plt.imshow(img.numpy())
        im_name = 'img_{}.png'.format(i)
        plt.savefig(os.path.join(save_dir, "Results", im_name), dpi=200, bbox_inches='tight')


def image_generation(save_folder=None):
    """input to the model is sampled input from mu= 1.0 and log_var = 0.3,
    it has the shape latent_dim + condition dim, meaning there is no additional condition.

    """
    batch = 16
    latent_dim = vae_net.latent_dim
    # sample both initial input and condition
    mu = torch.zeros(batch, latent_dim + condition_dim, 1, 1) + 1.0
    log_var = torch.zeros(batch, latent_dim + condition_dim, 1, 1) + 0.3
    # print(mu.shape)
    z_cond = vae_net.sample(mu.to(device), log_var.to(device))
    # print("zcond",z_cond.shape) #zcond torch.Size([128, 512, 1, 1])
    logits = vae_net.decoder(z_cond)
    generated = torch.sigmoid(logits)
    if save_folder:
        save_path = os.path.join(save_folder, "generation.png")
    else:
        save_path = os.path.join(result_folder, "generation.png")
    # vutils.save_image(generated, save_path)
    # print("save image at", save_path)
    save_image_grid(generated, save_path)
    print("save image at", save_path)


def image_generation_ones(save_folder=None):
    batch = 16
    latent_dim = vae_net.latent_dim
    # sample both initial input and condition
    # mu = torch.zeros(batch, latent_dim, 1, 1) + 1.0
    # log_var = torch.zeros(batch, latent_dim, 1, 1) + 0.3
    # trained_mu, trained_logvar
    # print(mu.shape)
    # zero_tensor = torch.zeros(batch, condition_dim, 1, 1).to(device)
    ones_tensor = torch.ones(batch, condition_dim, 1, 1).to(device)

    z_mean = 1.0
    z_log_var = 0.3

    eps = torch.randn(batch, latent_dim, 1, 1)
    z = z_mean + math.exp(z_log_var * .5) * eps
    z = z.to(device)

    # net mu [112, 128, 1, 1]
    # mu = vae_net.mu[0:batch]
    # log_var = vae_net.log_var[0:batch]
    # z = vae_net.sample(mu.to(device), log_var.to(device))

    z_cond = torch.cat((z, ones_tensor), dim=1)
    # print("zcond",z_cond.shape) #zcond torch.Size([128, 512, 1, 1])
    logits = vae_net.decoder(z_cond)
    generated = torch.sigmoid(logits)

    if save_folder:
        save_path = os.path.join(save_folder, "generation_conditioned_with_zero.png")
    else:
        save_path = os.path.join(result_folder, "generation_conditioned_with_zero.png")

    # vutils.save_image(generated, save_path)
    # print("save image at", save_path)

    save_image_grid(generated, save_path)
    print("save image at", save_path)


def image_generation_with_condition(test_labels, save_folder=None):
    batch = test_labels.shape[0]

    latent_dim = vae_net.latent_dim
    # sample both initial input and condition
    # mu = torch.zeros(batch, latent_dim, 1, 1) + 1.0
    # log_var = torch.zeros(batch, latent_dim, 1, 1) + 0.3
    #
    # # z = [16, 3, 64, 64]
    # z = vae_net.sample(mu.to(device), log_var.to(device))

    z_mean = 1.0
    z_log_var = 0.3

    # eps = torch.randn(batch, latent_dim, 1, 1)
    # torch.normal(mean=0.5, std=torch.arange(1., 6.))
    eps = torch.normal(mean=0.0, std=1.0, size=(batch, latent_dim, 1, 1))
    z = z_mean + math.exp(z_log_var * .5) * eps
    z = z.to(device)

    # vae_net.train()
    # with torch.no_grad():
    #     recon_data, mu, log_var = vae_net(test_images.to(device), test_labels.to(device))
    # z = vae_net.sample(mu.to(device), log_var.to(device))

    image_embed = test_labels.to(device)
    # image_embed = torch.reshape(image_embed, [-1, vae_net.condition_dim, 1, 1])
    # ones = torch.ones(z.shape[0], vae_net.condition_dim, 1, 1).to(device)
    # condition = ones * image_embed  # [16, 512, 1, 1]
    condition = torch.reshape(image_embed, [-1, vae_net.condition_dim, 1, 1])

    z_cond = torch.cat((z, condition), dim=1)

    # print("zcond",z_cond.shape) #zcond torch.Size([128, 512, 1, 1])
    logits = vae_net.decoder(z_cond)
    generated = torch.sigmoid(logits)
    if save_folder:
        folder = save_folder
    else:
        folder = result_folder

    save_path = os.path.join(folder, "generation_with_cond.png")
    # vutils.save_image(generated, save_path)
    # print("save image at", save_path)
    save_image_grid(generated, save_path)
    print("save image at", save_path)

    save_path_ori = os.path.join(folder, "generation_with_cond_ori.png")
    # vutils.save_image(test_images, save_path_ori)
    # print("save image at", save_path_ori)
    save_image_grid(test_images, save_path_ori)
    print("save image at", save_path_ori)


def image_generation_clip(target_attr=None, save_folder=None):
    """
    Generates and plots a batch of images with specific attributes (if given).

    - list target_attr : list of desired attributes [default None]
    """
    global clip_model
    if clip_model is None:
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model.cuda().eval()
        print("Load clip")
    batch = 16
    # Vector of user-defined attributes.
    if target_attr:

        text = clip.tokenize([target_attr]).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        # labels = text_features.repeat_interleave(batch, dim=0).to(device)
        labels = text_features.repeat(batch, 1)

        # labels_ = np.tile(text_features.cpu(), reps=[batch, 1])
        # print(labels == labels_ )
        print("Generation with attributes: ", target_attr)

    # Vector of attributes taken from the test set.
    else:
        # batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv')
        labels = test_labels.to(device)
        print("Generation with fixed attributes.")
        target_attr = "no attribute given"

    # vae_net.train()
    latent_dim = vae_net.latent_dim
    # mu = torch.zeros(batch, latent_dim, 1, 1) + 1.0
    # log_var = torch.zeros(batch, latent_dim, 1, 1) + 0.3
    # # print(mu.shape)
    # image_embed = torch.reshape(labels, [-1, 512, 1, 1])
    # ones = torch.ones(batch, 512, 1, 1).to(device)
    # condition = ones * image_embed  # [16, 32, 64, 64]
    # z = vae_net.sample(mu.to(device), log_var.to(device))  ####
    #
    condition = torch.reshape(labels, [batch, vae_net.condition_dim, 1, 1])

    z_mean = 1.0
    z_log_var = 0.3

    # eps = torch.randn(batch, latent_dim, 1, 1)
    # torch.normal(mean=0.5, std=torch.arange(1., 6.))
    eps = torch.normal(mean=0.0, std=1.0, size=(batch, latent_dim, 1, 1))
    z = z_mean + math.exp(z_log_var * .5) * eps
    z = z.to(device)

    # print("z", z.shape)
    # print("condition", condition.shape)
    z_cond = torch.cat((z, condition), dim=1)
    logits = vae_net.decoder(z_cond)
    generated = torch.sigmoid(logits)
    prompt = target_attr.replace(' ', '_')
    save_path = os.path.join(save_folder, "generation_" + prompt + ".png")

    # vutils.save_image(generated, save_path)
    # print("save image at", save_path)
    save_image_grid(generated, save_path)
    print("save image at", save_path)


def reconstruct_images(save_folder=None):
    recon_data, mu, var = vae_net(test_images.to(device), test_labels.to(device))
    recon_images = torch.cat((torch.sigmoid(recon_data).cpu(), test_images.cpu()), 2)

    if save_folder:
        save_path = os.path.join(save_folder, "recon.png")
    else:
        save_path = os.path.join(result_folder, "recon.png")
    # vutils.save_image(recon_images, save_path)
    # print("save image at", save_path)
    save_image_grid(recon_images, save_path)
    print("save image at", save_path)


def attribute_manipulation(target_attr=None, image_embed_factor=0.5, save_folder=None):
    global clip_model
    if clip_model is None:
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model.cuda().eval()

    reconstructed_images = []
    modified_images = []
    new_attr_factor = 1 - image_embed_factor
    for i in range(test_images.shape[0]):
        img = test_images[i][np.newaxis, ...]  # [1, 3, 64, 64]
        label = test_labels[i][np.newaxis, ...]  # [1, 512]
        vae_net.eval()
        recon_data, img_z, var = vae_net(img.to(device), label.to(device))
        # print("img_z", img_z.shape) # [1, 128, 1, 1]
        # print("..recon_data", recon_data.shape) #[1, 3, 64, 64]
        # recon_im = recon_data.cpu().detach().numpy()[0, :, :, :] # (3, 64, 64)
        recon_im = torch.sigmoid(recon_data)
        recon_im = recon_im.cpu().detach()[0, :, :, :]
        # print("recon_data.numpy()[i, :, :, :]", recon_im.shape)
        reconstructed_images.append(recon_im)

        if target_attr is None:
            # modified_label = np.expand_dims(test_labels[i], axis=0) # (1, 512)
            modified_label = label
            modified_label = modified_label.unsqueeze(2).unsqueeze(3)
            # print("modi label", modified_label.shape)
        else:
            text = clip.tokenize([target_attr]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text)
            modified_label = (label.to(device) * image_embed_factor) + (
                    text_features * new_attr_factor)  # (1, 512) #.cpu().detach().numpy() [1, 512]

            modified_label = modified_label.unsqueeze(2).unsqueeze(3)  # .unsqueeze(modified_label, 3)
            # print("modi label", modified_label.shape)

        z_cond = torch.cat((img_z, modified_label), dim=1)
        logits = vae_net.decoder(z_cond)
        generated = torch.sigmoid(logits)
        # modified_images.append(generated.detach().cpu().numpy()[0, :, :, :])
        modified_images.append(generated.detach().cpu()[0, :, :, :])
        # break
    # reconstructed_images = np.asarray(reconstructed_images, dtype='float32')
    # modified_images = np.asarray(modified_images, dtype='float32')

    print("modified_images", modified_images[0].shape)
    str_target_attr = str(target_attr).replace(' ', '_')
    str_target_attr_factors = f"{str_target_attr}_{image_embed_factor}_{new_attr_factor}"
    file_name = f"modified_images_{str_target_attr_factors}.png"

    if save_folder:
        save_path = os.path.join(save_folder, file_name)
    else:
        save_path = os.path.join(result_folder, file_name)
    # vutils.save_image(modified_images, save_path)
    save_image_grid(modified_images, save_path)
    print("save image at", save_path)

    file_name = f"recon_images_{str_target_attr_factors}.png"
    if save_folder:
        save_path = os.path.join(save_folder, file_name)
    else:
        save_path = os.path.join(result_folder, file_name)
    # reconstructed_images = np.asarray(reconstructed_images)
    # print("reconstructed_images",reconstructed_images.shape)
    # vutils.save_image(reconstructed_images, save_path)
    save_image_grid(reconstructed_images, save_path)
    print("save image at", save_path)


# ---------------------------------------------------------------
def train():
    loss_log = []

    # save log
    with open(os.path.join(result_folder, "params.txt"), "w") as f:
        f.write(f"epoch = {num_epoch}\n")
        f.write(f"learning_rate = {lr}\n")
        f.write(f"train_size = {train_size}\n")
        f.write(f"batch_size = {batch_size} \n")
        f.write(f"label_dim = {condition_dim}\n")
        f.write(f"image_size = {image_size}\n")
        f.write(f"latent_dim = {latent_dim}\n")
        f.write(f"beta = {beta}\n")
        f.write(f"model checkpoint = {model_save_path}\n\n")

    for epoch in trange(start_epoch, num_epoch, leave=False):
        start_time_epoch = time.perf_counter()
        lr_Linear(num_epoch, epoch, lr)
        vae_net.train()
        for i, (images, condition) in enumerate(tqdm(trainloader, leave=False)):
            images = images.to(device)
            condition = condition.to(device)  # [batch, 512]
            # recon_data = [batch, 3 + 512, 64, 64]
            recon_data, mu, logvar = vae_net(images, condition)
            trained_mu, trained_logvar = mu, logvar
            # VAE loss
            loss = vae_loss(recon_data, images, mu, logvar)

            # Perception loss
            loss += feature_extractor(torch.cat((torch.sigmoid(recon_data), images), 0))

            loss_log.append(loss.item())
            vae_net.zero_grad()
            loss.backward()
            optimizer.step()

        # In eval mode the model will use mu as the encoding instead of sampling from the distribution
        print("epoch", epoch)
        exec_time_epoch = time.perf_counter() - start_time_epoch

        print(f"time epoch = {exec_time_epoch} sec ({exec_time_epoch / 60.0} min )\n")
        with open(os.path.join(result_folder, "params.txt"), "a") as f:
            f.write(f"\nepoch {epoch}, time epoch = {exec_time_epoch} sec ({exec_time_epoch / 60.0} min )\n")

        result_folder_epoch = os.path.join(result_folder, f"{epoch}")
        if not os.path.exists(result_folder_epoch):
            os.mkdir(result_folder_epoch)

        vae_net.eval()
        with torch.no_grad():
            recon_data, _, _ = vae_net(test_images.to(device), test_labels.to(device))
            images = torch.cat((torch.sigmoid(recon_data.cpu()), test_images), 2)
            save_path = os.path.join(result_folder_epoch, "recon" + "_" + str(epoch) + ".png")
            # save_path = "%s/%s/%s_%d_%d.png" % (save_dir, "Results", model_name, image_size, epoch)
            # print(images.shape)  # [128, 3, 128, 64]
            print("save image at", save_path)
            vutils.save_image(images, save_path)
            image_generation_clip(target_attr="sad", save_folder=result_folder_epoch)
            image_generation_clip(target_attr="wearing glasses", save_folder=result_folder_epoch)
            image_generation()

        # Save a checkpoint
        if (epoch + 1) % save_model_every == 0:
            print("true")
            model_save_path_epoch = os.path.join(save_dir, "Models", f"{model_name}_epoch{epoch}.pt")
            torch.save(vae_net, model_save_path_epoch)
            # torch.save({
            #     'epoch': epoch,
            #     'loss_log': loss_log,
            #     'model_state_dict': vae_net.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict()
            #
            # }, model_save_path)
            # torch.save(vae_net, model_save_path)
            print("Save checkpoint at", model_save_path_epoch)

    exec_time_total = time.perf_counter() - start_time_total
    print(f"time total = {exec_time_total} sec ({exec_time_total / 60.0} min )\n")
    with open(os.path.join(result_folder, "params.txt"), "a") as f:
        f.write(f"\ntime total = {exec_time_total} sec ({exec_time_total / 60.0} min )\n")


# ---------------------------------------------------------------
if __name__ == "__main__":
    if run_train:
        train()
    attribute_manipulation(target_attr="wear reading glasses", image_embed_factor=0.5, save_folder=result_folder)
    image_generation_clip(target_attr="old woman", save_folder=result_folder)
    # image_generation_clip(target_attr="funny", save_path=result_folder)
    # image_generation_clip(target_attr="a woman with a baseball cap", save_path=result_folder)
    # image_generation_clip(target_attr="a woman with a bang", save_path=result_folder)
    # image_generation_clip(target_attr="crying", save_path=result_folder)
    # image_generation_clip(target_attr="very sad", save_path=result_folder)
    # image_generation_clip(target_attr="a photo of a sad person", save_path=result_folder)
    # image_generation_clip(target_attr="smiling", save_path=result_folder)
    # image_generation_clip(target_attr="smiling woman", save_path=result_folder)
    # image_generation_clip(target_attr="baby", save_path=result_folder)
    # image_generation_clip(target_attr="dog", save_path=result_folder)
    # image_generation_clip(target_attr="hand", save_path=result_folder)
    # image_generation_clip(target_attr="a man", save_path=result_folder)
    # image_generation_clip(target_attr="asian", save_path=result_folder)
    # image_generation_clip(target_attr="a red hair woman", save_path=result_folder)
    # image_generation_clip(target_attr="wearing glasses", save_path=result_folder)
    # image_generation_clip(target_attr="a photo of a person wearing glasses", save_path=result_folder)
    #
    reconstruct_images(save_folder=result_folder)
    image_generation_with_condition(test_labels, save_folder=result_folder)
    image_generation(save_folder=result_folder)
    image_generation_ones(save_folder=result_folder)
