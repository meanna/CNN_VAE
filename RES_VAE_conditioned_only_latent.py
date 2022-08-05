# no condition in encoder, only in latent space before encoding

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

use_cuda = torch.cuda.is_available()
GPU_indx = 0
device = torch.device(GPU_indx if use_cuda else "cpu")


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, scale=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.UpNN = nn.Upsample(scale_factor=scale, mode="nearest")

    def forward(self, x):
        skip = self.conv3(self.UpNN(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x


class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """

    def __init__(self, channels, ch=64, z=512):
        super(Encoder, self).__init__()
        self.conv1 = ResDown(channels, ch)  # 64
        self.conv2 = ResDown(ch, 2 * ch)  # 32
        self.conv3 = ResDown(2 * ch, 4 * ch)  # 16
        self.conv4 = ResDown(4 * ch, 8 * ch)  # 8
        self.conv5 = ResDown(8 * ch, 8 * ch)  # 4
        self.conv_mu = nn.Conv2d(8 * ch, z, 2, 2)  # 2
        self.conv_log_var = nn.Conv2d(8 * ch, z, 2, 2)  # 2

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.training:
            mu = self.conv_mu(x)
            log_var = self.conv_log_var(x)
            x = self.sample(mu, log_var)
        else:
            mu = self.conv_mu(x)
            x = mu
            log_var = None

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, z=512):
        super(Decoder, self).__init__()
        self.conv1 = ResUp(z, ch * 8)
        self.conv2 = ResUp(ch * 8, ch * 8)
        self.conv3 = ResUp(ch * 8, ch * 4)
        self.conv4 = ResUp(ch * 4, ch * 2)
        self.conv5 = ResUp(ch * 2, ch)
        self.conv6 = ResUp(ch, ch // 2)
        self.conv7 = nn.Conv2d(ch // 2, 3, 3, 1, 1)  # the second dim is output image channel = 3

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """

    def __init__(self, channel_in, ch=64, z=512, condition_dim=512):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        self.latent_dim = z
        channel_in = 3
        self.condition_dim = condition_dim
        self.encoder = Encoder(channel_in, ch=ch, z=z)
        self.decoder = Decoder(channel_in, ch=ch, z=z + self.condition_dim)

    def forward(self, x, image_embed):

        # image_embed = [batch, embed dim]
        #image_embed = torch.randn(x.shape[0], self.condition_dim).to(device)
        image_embed = torch.reshape(image_embed, [-1, self.condition_dim, 1, 1])
        # ones = torch.ones(x.shape[0], self.condition_dim, 64, 64).to(device)
        # condition = ones * image_embed  # [16, 32, 64, 64]
        # # x = torch.Size([128, 3, 64, 64])
        # x = torch.cat((x, condition), dim=1)

        # encoding  =[batch, latent=128, 1, 1]
        encoding, mu, log_var = self.encoder(x)

        # encoding = [batch, latent + image_embed, 1, 1]
        encoding = torch.cat((encoding, image_embed), dim=1)
        recon = self.decoder(encoding)
        return recon, mu, log_var

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterization(self, z_mean, z_log_var):
        """ Performs the reparameterization trick"""

        eps = torch.randn(z_mean.shape[0], self.latent_dim, 1, 1).to(device)
        # z_mean = torch.randn(32, 128,1,1)
        # z_log_var = torch.randn(32, 128,1,1)
        z = z_mean + torch.exp(z_log_var * .5) * eps
        # z_cond = torch.cat([z, input_label], dim=1)
        return z
