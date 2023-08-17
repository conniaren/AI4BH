import vcf_data_loader
import pytorch_lightning as pl
import torch 
from torch import nn 
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

CUDA_LAUNCH_BLOCKING=1

class autoencoder(pl.LightningModule):
    def __init__(self, input_size, base_layer_size = 512, lr = 1e-3, mnist = False):
        super().__init__()
        self.mnist = mnist
        self.input_size = input_size
        self.encoder = nn.Sequential(nn.Linear(input_size, base_layer_size), 
                                     nn.BatchNorm1d(base_layer_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(base_layer_size, base_layer_size//2),
                                     nn.BatchNorm1d(base_layer_size//2),
                                     nn.LeakyReLU(),
                                     nn.Linear(base_layer_size//2, base_layer_size//4),
                                     nn.BatchNorm1d(base_layer_size//4),
                                     nn.LeakyReLU(),
                                     nn.Linear(base_layer_size//4, base_layer_size//6))
        self.decoder = nn.Sequential(nn.Linear(base_layer_size//6, base_layer_size//4),
                                     nn.BatchNorm1d(base_layer_size//4),
                                     nn.LeakyReLU(),
                                     nn.Linear(base_layer_size//4, base_layer_size//2),
                                     nn.BatchNorm1d(base_layer_size//2),
                                     nn.LeakyReLU(),
                                     nn.Linear(base_layer_size//2, base_layer_size),
                                     nn.BatchNorm1d(base_layer_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(base_layer_size,input_size*2),
                                     nn.Unflatten(1,(2,input_size)),
                                     nn.Softmax(dim = 1))
        self.lr = lr
        self.save_hyperparameters()
    
    def forward(self, x):
        x = x.to(torch.device("cuda"))
        if self.mnist: 
            x = x.view(x.size(0),-1)
        x_tilda = self.encoder(x)
        y = self.decoder(x_tilda)
        top_p, top_class = y.topk(1, dim = 1)
        return top_class
    
    def training_step(self, batch, i):
        if self.mnist: 
            batch, label = batch
            batch = batch.view(batch.size(0),-1).to(torch.device("cuda"))
        z = self.encoder (batch)
        x_hat = self.decoder(z)
        regularisation = 0
        for param in self.parameters():
            if len(param.shape)>1:
                regularisation += torch.mean(torch.square(param))
        ce_loss = nn.CrossEntropyLoss()
        batch = batch.type(torch.LongTensor).to(torch.device("cuda"))
        loss = ce_loss(x_hat,batch) + 1e-4*regularisation
        self.log("Training_error", loss, on_epoch = True, on_step = False, prog_bar = True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = self.lr)
    

vcf = vcf_data_loader.FixedSizeVCFChunks("ReferencePanel_v1_highQual_MAF0.005_filtered.vcf.gz", max_snps_per_chunk=690, create = True)
data_o = vcf.get_tensor_for_chunk_id(0)

print(torch.unique(data_o, return_counts = True))

#MNIST dataset
def quantize(image:torch.Tensor)->torch.Tensor:
    for i in range(image.size()[1]):
        for j in range(image.size()[2]):
            if image[:,i,j] <0.5:
                image[:,i,j] = 0
            else:
                image[:,i,j] = 1
    
    return image 


transform  = transforms.Compose([transforms.ToTensor(), transforms.Lambda(quantize)])
train_data = datasets.MNIST(root='data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)

epochs = 10

#Data processing: replace -1 with medians
torch.manual_seed(555)
nump = data_o.numpy()
df = pd.DataFrame(nump)
df[df == -1] = np.nan 
df = df.fillna(df.median(axis=0))
data_o = torch.tensor(df.values).to(device = 'cuda')

print(torch.unique(data_o, return_counts = True))


print("-----------------------------------")
dataloader = DataLoader(data_o, batch_size=100, num_workers=8)
model = autoencoder(690)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, num_workers=64, pin_memory = True, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, num_workers=64, pin_memory = True, shuffle = True)

model_mnist = autoencoder(784, mnist=True).to(torch.device("cuda"))

dataiter = iter(train_loader)
images, labels = next(dataiter)
img_flat = images.view(images.size(0), -1)

print(torch.unique(img_flat, return_counts = True))

'''
trainer = pl.Trainer(logger = pl.loggers.TensorBoardLogger("tb_logs", name = "autoencoder_model"),
                     log_every_n_steps = 1, 
                     accelerator= "gpu",
                     max_epochs = epochs,
                     devices = -1)
'''

#trainer.fit(model_mnist,train_loader)

#path = f"./model-mnist.pth"
#torch.save(model_mnist.state_dict(), path)

import matplotlib.pyplot as plt 

model_mnist.load_state_dict(torch.load(f"./model-mnist.pth"))



dataiter = iter(test_loader)
images, labels = next(dataiter)

images_flatten = images.view(images.size(0), -1)
# get sample outputs
output = model_mnist(images_flatten)
# prep images for display
images = images.numpy()

# output is resized into a batch of images
output = output.view(20, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().cpu().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
trainer.fit(model_mnist,train_loader)
