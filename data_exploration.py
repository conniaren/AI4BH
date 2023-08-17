import vcf_data_loader
import pytorch_lightning as pl
import torch 
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset

class autoencoder(pl.LightningModule):
    def __init__(self, input_size, base_layer_size = 512, lr = 1e-3):
        super().__init__()
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
                                     nn.Linear(base_layer_size,input_size*3),
                                     nn.Softmax())
        self.lr = lr
    
    def forward(self, x):
        x_tilda = self.encoder(x)
        y = self.decoder(x_tilda)
        y = y.view(x.size(0), 3, self.input_size)
        top_p, top_class = y.topk(1, dim = 1)
        return top_class
    
    def training_step(self, batch, i):
        z = self.encoder (batch)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(batch.size(0), 3, self.input_size)
        regularisation = 0
        for param in self.parameters():
            if len(param.shape)>1:
                regularisation += torch.mean(torch.square(param))
        ce_loss = nn.CrossEntropyLoss()
        loss = ce_loss(batch, x_hat) + 1e-4*regularisation
        self.log("Training_error", loss, on_epoch = True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = self.lr)
    

vcf = vcf_data_loader.FixedSizeVCFChunks("ReferencePanel_v1_highQual_MAF0.005_filtered.vcf.gz", max_snps_per_chunk=690, create = True)
data_o = vcf.get_tensor_for_chunk_id(0)


print(torch.unique(data_o, return_counts = True))
'''
epochs = 1

torch.manual_seed(555)

print("-----------------------------------")
dataloader = DataLoader(data_o, batch_size=100, num_workers=8)
model = autoencoder(690)
trainer = pl.Trainer(logger = pl.loggers.TensorBoardLogger("tb_logs", name = "autoencoder_model"),
                     log_every_n_steps = 1, 
                     accelerator= "cpu",
                     max_epochs = epochs)

trainer.fit(model,dataloader)
'''
