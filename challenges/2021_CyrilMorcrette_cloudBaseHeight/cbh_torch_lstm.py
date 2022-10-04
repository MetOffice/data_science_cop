import torch
import pytorch_lightning as pl


# define RNN
class CloudBaseLSTM(pl.LightningModule):
    def __init__(self, input_size, lstm_layers, lstm_hidden_size, output_size, height_dimension, embed_size, BILSTM=True, batch_first=False, lr=2e-3, do_base_label_fit=True, do_vol_fit=False):
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
        
        self.do_vol_fit = do_vol_fit
        if do_vol_fit:
            self.loss_fn_vol = torch.nn.MSELoss()
            self.relu = torch.nn.ReLU()

        self.do_base_label_fit = do_base_label_fit
        if do_base_label_fit:
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
        
        nn_out = None # initialize for clarity
        relu_out = None
        # apply ReLU
        if self.do_vol_fit:
            lstm_out = self.relu(lstm_out)
            relu_out = lstm_out # save for potential double loss
        
        if self.do_base_label_fit:
            # apply linear layer for base prediction
            nn_out = self.linearCap(lstm_out)
            
        return nn_out, relu_out # return both the nn_out and the lstm out for loss calculations
        
    
    def generic_model_step(self, batch, batch_idx, str_of_step_name):
        # print("Start step")
        
         #### #### #### WARNING MAY CAUSE SOME WEIRD OBJECT ORIENTED RELATED BEHAVIOUR I AM UNAWARE ABOUT AND NOT WORK #### #### ####
            
        loss_1, loss_2 = 0, 0
        
        base_pred, vol_pred = self(batch['x'], batch['height_vector'])
        
        if self.do_vol_fit:
            loss_1 = self.loss_fn_vol(vol_pred, batch['cloud_volume_target'])
            loss_1_name = str_of_step_name + ' volume loss component'
            self.log(loss_1_name, loss_1)
            
        if self.do_base_label_fit:
            base_targets = batch['cloud_base_target']
            base_targets = torch.flatten(base_targets)
            # print('basetarg shape:',base_targets.shape)
            # print('basepred shape:',base_pred.shape)
            loss_2 = self.loss_fn_base(base_pred, base_targets)
            loss_name = str_of_step_name + ' base height loss component'
            self.log(loss_name, loss_2)
             
        loss = (loss_1*40) + loss_2 # 40 adjusts for differences in numerical values produced by loss function
        
        #log to tensorboard
        self.log((str_of_step_name + ' loss'), loss)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        
        return self.generic_model_step(batch, batch_idx, 'training')
    
    def validation_step(self, batch, batch_idx):
        
        return self.generic_model_step(batch, batch_idx, 'validation')
    
    def test_step(self, batch, batch_idx):
        
        return self.generic_model_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.lr)
        
        return optim