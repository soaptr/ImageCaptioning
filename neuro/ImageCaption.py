import torch
import torch.nn as nn
# import torchvision
import cv2
import numpy as np
# import os
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
# from torchvision.transforms import Resize, RandomCrop, InterpolationMode
import youtokentome as yttm
import urllib


IMAGE_SIZE = 384

val_transforms = albu.Compose(
    [
        albu.SmallestMaxSize(IMAGE_SIZE),
        albu.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
        albu.Normalize(),
        ToTensorV2(),
    ]
)


class EncoderCNN(nn.Module):

    def __init__(self, embed_size, train_CNN=False, dropout=0):
        super(EncoderCNN, self).__init__()
        self.encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = train_CNN
        dummy_input = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        in_features = self.encoder(dummy_input).shape[1]
        self.encoder_linear = nn.Linear(in_features, embed_size)

    def forward(self, images):
        features = self.encoder(images)
        features = features.view(features.shape[0], -1)
        features = self.encoder_linear(features)
        return features


class DecoderTransformer(nn.Module):

    def __init__(self, embed_size, dim_feedforward, vocab_size, num_heads,
                 num_layers, device, dropout=0.1, max_length=20):
        super(DecoderTransformer, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.positional_embed = nn.Embedding(max_length, embed_size)
        decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads, dim_feedforward,
                                                   dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.max_length = max_length
        self.embed_size = embed_size
        self.device = device

    def forward(self, features, caption, target_pad_mask):
        batch_size, sequence_length = caption.shape[0], caption.shape[1]

        scale = torch.sqrt(torch.tensor([self.embed_size])).to(self.device)
        x = self.token_embed(caption) * scale
        position = (
            torch.arange(0, sequence_length)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(self.device)
        )
        x += self.positional_embed(position)  # [batch, seq, emb_size]

        features = features.unsqueeze(1)  # [batch, 1, emb_size]
        target_subsequent_mask = (
            nn.Transformer()
            .generate_square_subsequent_mask(x.shape[1])
            .to(self.device)
        )
        x = self.decoder(
            x,
            features,
            tgt_mask=target_subsequent_mask,
            tgt_key_padding_mask=target_pad_mask,
        )
        out = self.linear(x)
        return out


class ImageCaptioningModelTransformer(nn.Module):

    def __init__(self, embed_size, dim_feedforward, vocab_size, num_heads, num_layers,
                 device, train_CNN=False, dropout=0.1, max_length=20):
        super(ImageCaptioningModelTransformer, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size, train_CNN)
        self.decoderTransformer = DecoderTransformer(embed_size, dim_feedforward, vocab_size,
                                                     num_heads, num_layers, device, dropout, max_length)
        self.device = device
        if device == torch.device('cuda'):
            self.cuda()
        else:
            self.cpu()

    def forward(self, images, captions, attn_mask):
        features = self.encoderCNN(images)
        outputs = self.decoderTransformer(features, captions, attn_mask)
        return outputs


def softmax(x, temperature):
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)


def caption_image_transformer(model, image, tokenizer, temp=0, max_length=20):
    result_caption = []

    image = image.to(model.device).unsqueeze(0)

    target_indexes = [tokenizer.subword_to_id('<BOS>')] + [tokenizer.subword_to_id('<PAD>')] * (max_length - 1)
    for i in range(max_length - 1):
        caption = torch.LongTensor(target_indexes).unsqueeze(0)
        mask = torch.zeros((1, max_length), dtype=torch.bool)
        mask[:, i+1:] = True

        with torch.no_grad():
            output = model(image, caption.to(model.device), mask.to(model.device)).squeeze(0)  # [seq_len, vocab_size]

        output = output[i]

        if temp <= 0:
            predicted = output.argmax(0)
        else:
            output = softmax(output, temp)
            predicted = torch.multinomial(output, 1).squeeze(0)
        predicted = predicted.item()
        if predicted == tokenizer.subword_to_id('<EOS>'):
            break
        result_caption.append(predicted)
        target_indexes[i + 1] = predicted

    return tokenizer.decode(result_caption)[0]


def load():
    tokenizer = yttm.BPE('neuro/models/BPE_model.bin')
    vocabulary = tokenizer.vocab()
    embed_size = 800
    dim_feedforward = 2048
    vocab_size = len(vocabulary)
    num_heads = 8
    num_layers = 1
    train_CNN = False
    dropout = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageCaptioningModelTransformer(embed_size, dim_feedforward, vocab_size, num_heads,
                                            num_layers, device, train_CNN, dropout)
    model.load_state_dict(torch.load('neuro/models/last-transformer-model-2048-1l-8h-v2.pt', map_location=device))
    model.eval()
    return model, tokenizer


def transform_image(file_path):
    req = urllib.request.urlopen(file_path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = val_transforms(image=img)["image"]
    return img
