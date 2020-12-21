import torch.nn as nn
import torch.nn.functional as func


#This contains both encoder and decoder classes for autoencoder
class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.ngpu=ngpu
        self.encoder_pt1 = nn.Sequential(
            nn.Conv1d(1, 3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(3, 5, 3, stride=1, padding=1),
        )
        self.encoder_pt2 = nn.Sequential(
            nn.Linear(35, 20),
            nn.Linear(20, 10)
        )
        self.decoder_pt1 = nn.Sequential(
            nn.Linear(10, 35),
            nn.Linear(35, 56)
        )
        self.decoder_pt2 = nn.Sequential(
            nn.ConvTranspose1d(4, 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(2, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        encoded_mid = self.encoder_pt1(input).view(-1, 1, 35)
        embedding = self.encoder_pt2(encoded_mid)

        decoded_mid = self.decoder_pt1(embedding).view(-1, 4, 14)
        reconstructed = self.decoder_pt2(decoded_mid)
        return reconstructed



"""
TWO Layers With Pooling
"""
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder_pt1 = nn.Sequential(
#             nn.Conv1d(1, 4, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(4, 14, 3, stride=2, padding=1),
#         )
#         self.encoder_pt2 = nn.Sequential(
#             nn.Linear(28, 14),
#             nn.Linear(14, 4)
#         )
#         self.decoder_pt1 = nn.Sequential(
#             nn.Linear(4, 14),
#             nn.Linear(14, 28)
#         )
#         self.decoder_pt2 = nn.Sequential(
#             nn.ConvTranspose1d(7, 4, 3, stride=2, padding=1),
#             nn.ReLU(),
#             # nn.MaxUnpool1d(2),
#             nn.ConvTranspose1d(4, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         mid = self.encoder_pt1(input)
#         encoded_mid = mid.view(-1,1,28)
#         embedding = self.encoder_pt2(encoded_mid)

#         mid_d = self.decoder_pt1(embedding)
#         decoded_mid_d = mid_d.view(-1,7,4)
#         reconstructed = self.decoder_pt2(decoded_mid_d)
#         return reconstructed


"""
ONE LAYER
"""
# #This contains both encoder and decoder classes for autoencoder
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder_pt1 = nn.Sequential(
#             nn.Conv1d(1, 4, 3, stride=2, padding=1),
#             # nn.ReLU(),
#             # nn.Conv1d(4, 14, 3, stride=2, padding=1),
#         )
#         self.encoder_pt2 = nn.Sequential(
#             # nn.Linear(56, 20),
#             # nn.Linear(20, 4)
#             nn.Linear(28, 14),
#             nn.Linear(14,4)
#         )
#         self.decoder_pt1 = nn.Sequential(
#             # nn.Linear(4, 20),
#             # nn.Linear(20, 56)
#             nn.Linear(4,14),
#             nn.Linear(14,28)
#         )
#         self.decoder_pt2 = nn.Sequential(
#             # nn.ConvTranspose1d(14, 4, 3, stride=2, padding=1),
#             # nn.ReLU(),
#             nn.ConvTranspose1d(4, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         # encoded_mid = self.encoder_pt1(input).view(-1, 1, 56)
#         mid = self.encoder_pt1(input)
#         encoded_mid = mid.view(-1, 1, 28)
#         embedding = self.encoder_pt2(encoded_mid)

#         # decoded_mid = self.decoder_pt1(embedding).view(-1, 14, 4)
#         decoded_mid = self.decoder_pt1(embedding).view(-1, 4, 7)
#         reconstructed = self.decoder_pt2(decoded_mid)
#         return reconstructed


# """
# THREE LAYERS
# """
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder_pt1 = nn.Sequential(
#             nn.Conv1d(1, 4, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(4, 8, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(8, 14, 3, stride=2, padding=1)
#         )
#         self.encoder_pt2 = nn.Sequential(
#             nn.Linear(28, 14),
#             nn.Linear(14, 4)
#         )
#         self.decoder_pt1 = nn.Sequential(
#             nn.Linear(4, 14),
#             nn.Linear(14, 56)
#         )
#         self.decoder_pt2 = nn.Sequential(
#             nn.ConvTranspose1d(14, 8, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(8, 4, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(4, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         encoded_mid = self.encoder_pt1(input).view(-1, 1, 28)
#         embedding = self.encoder_pt2(encoded_mid)

#         decoded_mid = self.decoder_pt1(embedding).view(-1, 14, 4)
#         reconstructed = self.decoder_pt2(decoded_mid)
#         return reconstructed

# """
# THREE LAYERS ONE MAXPOOL
# """
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder_pt1 = nn.Sequential(
#             nn.Conv1d(1, 4, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(4, 8, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(8, 14, 3, stride=2, padding=1)
#         )
#         self.encoder_pt2 = nn.Sequential(
#             nn.Linear(14, 7),
#             nn.Linear(7, 4)
#         )
#         self.decoder_pt1 = nn.Sequential(
#             nn.Linear(4, 14),
#             nn.Linear(14, 28)
#         )
#         self.decoder_pt2 = nn.Sequential(
#             nn.ConvTranspose1d(7, 5, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(5, 3, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(3, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         encoded_mid = self.encoder_pt1(input).view(-1, 1, 14)
#         embedding = self.encoder_pt2(encoded_mid)

#         decoded_mid = self.decoder_pt1(embedding).view(-1, 7, 4)
#         reconstructed = self.decoder_pt2(decoded_mid)
#         return reconstructed
