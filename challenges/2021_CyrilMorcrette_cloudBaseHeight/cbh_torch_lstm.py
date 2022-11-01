import torch
import pytorch_lightning as pl


# define RNN
class CloudBaseLSTM(pl.LightningModule):
    def __init__(self, input_size, lstm_layers, lstm_hidden_size, output_size, height_dimension, embed_size, BILSTM=True, batch_first=False, lr=2e-3):
        super().__init__()
        
        self.LSTM = torch.nn.LSTM(
                                  input_size+embed_size, 
                                  lstm_hidden_size, 
                                  lstm_layers, 
                                  batch_first=batch_first, 
                                  bidirectional=BILSTM, 
                                  proj_size=output_size
                                 )
        
        self.batch_first = batch_first
        self.proj_size = output_size
        
        
        self.height_embedding = torch.nn.Embedding(height_dimension, embed_size)
        self.BILSTM = BILSTM
        self.lr = lr

        self.loss_fn_base = torch.nn.CrossEntropyLoss()
        self.linearCap = torch.nn.Linear(height_dimension, height_dimension)
        
        self.save_hyperparameters() # save hyperparameters for 
        
    def forward(self, x, height):
        
        #produce height embeds
        height_embeds = self.height_embedding(height)
        height_embeds = torch.flatten(height_embeds, start_dim=2)
        # print(height_embeds.size())
        
        #concat with feature vector
        x_and_height = torch.cat((x, height_embeds), 2)
        
        #send through LSTM
        lstm_out, _ = self.LSTM(x_and_height)
        # combine backward and forward LSTM outputs for each cell
        if(self.BILSTM):
            lstm_out = lstm_out[:,:,:self.proj_size] + lstm_out[:,:,self.proj_size:]
        # combinedLSTMOut = combinedLSTMOut / 2 # possibility, hyperparameter of combination method
        
        # flatten seq out (each cell produced 1 value for a height layer, so combine all cell outputs to a sequence of height layers for application of further nn layers)
        lstm_out = torch.flatten(lstm_out, start_dim=1)
        
        nn_out = self.linearCap(lstm_out)
            
        return nn_out, relu_out # return both the nn_out and the lstm out for loss calculations
    
    def training_step(self, batch, batch_idx):
        
        base_pred, vol_pred = self(batch['x'], batch['height_vector'])
        
            
        base_targets = batch['cloud_base_target']
        base_targets = torch.flatten(base_targets)
        loss = self.loss_fn_base(base_pred, base_targets)
        loss_name = str_of_step_name + ' base height loss component'
        self.log(loss_name, loss)
             
        #log
        self.log(("train" + ' loss'), loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        base_pred, vol_pred = self(batch['x'], batch['height_vector'])
        
            
        base_targets = batch['cloud_base_target']
        base_targets = torch.flatten(base_targets)
        loss = self.loss_fn_base(base_pred, base_targets)
        loss_name = str_of_step_name + ' base height loss component'
        self.log(loss_name, loss)
             
        #log
        self.log(("val" + ' loss'), loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        
        base_pred, vol_pred = self(batch['x'], batch['height_vector'])
        
            
        base_targets = batch['cloud_base_target']
        base_targets = torch.flatten(base_targets)
        loss = self.loss_fn_base(base_pred, base_targets)
        loss_name = str_of_step_name + ' base height loss component'
        self.log(loss_name, loss)
             
        #log
        self.log(("test" + ' loss'), loss)
        
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.lr)
        
        return optim