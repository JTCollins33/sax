import torch.nn as nn
import torch.nn.functional as func


#This contains both encoder and decoder classes for autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder_pt1 = nn.Sequential(
            nn.Conv1d(1, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(4, 14, 3, stride=2, padding=1),
        )
        self.encoder_pt2 = nn.Sequential(
            nn.Linear(56, 20),
            nn.Linear(20, 4)
        )
        self.decoder_pt1 = nn.Sequential(
            nn.Linear(4, 20),
            nn.Linear(20, 56)
        )
        self.decoder_pt2 = nn.Sequential(
            nn.ConvTranspose1d(14, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        encoded_mid = self.encoder_pt1(input).view(-1, 1, 56)
        embedding = self.encoder_pt2(encoded_mid)

        decoded_mid = self.decoder_pt1(embedding).view(-1, 14, 4)
        reconstructed = self.decoder_pt2(decoded_mid)
        return reconstructed