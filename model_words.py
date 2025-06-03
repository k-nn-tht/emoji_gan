import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, embed_size):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_size)
        self.embed_size = embed_size
        input_channels = channels_img + embed_size
        self.disc = nn.Sequential(
            # wrapped Conv2d with nn.utils.spectral_norm
            nn.utils.spectral_norm(
                nn.Conv2d(input_channels, features_d, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # wrapped Conv2d with nn.utils.spectral_norm
            nn.utils.spectral_norm(
                nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)
            ),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            # wrapped Conv2d with nn.utils.spectral_norm
            nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        label_embed = self.label_embedding(labels)
        label_map = label_embed[:, :, None, None].repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, label_map], dim=1)
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, embed_size):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_size)
        input_dim = channels_noise + embed_size
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(input_dim, features_g * 16, 4, 1, 0), # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1), # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1), # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels)  # (N, embed_size)
        x = torch.cat([noise.squeeze(), label_embed], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(x)



def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    num_classes = 114
    embed_size = 50  # or whatever you choose
    disc = Discriminator(in_channels, 8, num_classes, embed_size)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8, num_classes, embed_size)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()