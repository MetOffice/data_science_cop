import torch
import pytorch_lightning as pl


# define RNN
class CloudBaseLSTM(pl.LightningModule):
    def __init__(self, input_size, lstm_layers, lstm_hidden_size, output_size, height_dimension, embed_size, BILSTM=False, backward_lstm_differing_transitions=False, batch_first=False, lr=2e-3, skip_connection=False):
        super().__init__()
        
        self.LSTM_upward = torch.nn.LSTM(
                                  input_size+embed_size, 
                                  lstm_hidden_size, 
                                  lstm_layers, 
                                  batch_first=batch_first, 
                                  bidirectional=BILSTM, 
                                  proj_size=output_size
                                 )
        if backward_lstm_differing_transitions:
            self.LSTM_downward = torch.nn.LSTM(
                                      input_size+embed_size, 
                                      lstm_hidden_size, 
                                      lstm_layers, 
                                      batch_first=batch_first, 
                                      bidirectional=BILSTM, 
                                      proj_size=output_size
            )
        
        self.batch_first = batch_first
        self.proj_size = output_size
        self.backward_lstm_differing_transitions = backward_lstm_differing_transitions
        self.height_dim = height_dimension
        self.height_embedding = torch.nn.Embedding(height_dimension, embed_size)
        self.BILSTM = BILSTM
        self.lr = lr
        self.skip = skip_connection
        self.loss_fn_base = torch.nn.CrossEntropyLoss()
        if skip_connection:
            self.linearCap = torch.nn.Linear(height_dimension*(input_size+embed_size+output_size), height_dimension)
        else:
            self.linearCap = torch.nn.Linear(height_dimension, height_dimension)
            
        self.layer_norm = torch.nn.LayerNorm((input_size))
        
        self.save_hyperparameters() # save hyperparameters for 
        
    def forward(self, x):
        
        x = self.layer_norm(x)
        #produce height embeds
        if self.height_dim>0:
            height = torch.tensor(torch.arange(0,70)).repeat((len(x),1)).reshape(len(x),70,1)
            # print(height.size())
            height_embeds = self.height_embedding(height)
            # print(height_embeds.size())
            height_embeds = torch.flatten(height_embeds, start_dim=2)
            # print(height_embeds.size())
            x_and_height = torch.cat((x, height_embeds), 2)
            # print(x_and_height.size())
        else:
            x_and_height=x
        # print(height_embeds.size())
        
        #concat with feature vector
        
        
        #send through LSTM
        lstm_out, _ = self.LSTM_upward(x_and_height)
        if self.backward_lstm_differing_transitions:
            lstm_out_downward, _ = self.LSTM_downward(torch.fliplr(x_and_height))
            if(self.BILSTM):
                lstm_out_downward[:,:,:self.proj_size] = torch.fliplr(lstm_out_downward[:,:,:self.proj_size])
                lstm_out_downward[:,:,self.proj_size:] = torch.fliplr(lstm_out_downward[:,:,self.proj_size:])
            else:
                lstm_out_downward = torch.fliplr(lstm_out_downward)
                
            lstm_out = lstm_out + lstm_out_downward
        # combine backward and forward LSTM outputs for each cell (BILSTM)
        if(self.BILSTM):
            lstm_out = lstm_out[:,:,:self.proj_size] + lstm_out[:,:,self.proj_size:]
        # combinedLSTMOut = combinedLSTMOut / 2 # possibility, hyperparameter of combination method
        
        # flatten seq out (each cell produced 1 value for a height layer, so combine all cell outputs to a sequence of height layers for application of further nn layers)
        lstm_out = torch.flatten(lstm_out, start_dim=1)
        
        if self.skip:
            lstm_out = torch.cat((lstm_out, torch.flatten(x_and_height, start_dim=1)), 1)
            
        
        nn_out = self.linearCap(lstm_out)
            
        return nn_out
    
    def training_step(self, batch, batch_idx):
        
        inputs = batch[0]
        targets = batch[1]
        targets = torch.flatten(targets)

        predictions = self(inputs)

        loss = self.loss_fn_base(predictions, targets)
        
        # log to mlflow
        self.logger.log_metrics({"Train loss" : loss}, step=self.global_step)
        # log to pl for checkpointing
        self.log('Train loss', loss)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        inputs = batch[0]
        targets = batch[1]
        targets = torch.flatten(targets)

        predictions = self(inputs)

        loss = self.loss_fn_base(predictions, targets)
        
        # log to mlflow
        self.logger.log_metrics({"Val loss" : loss}, step=self.global_step)
        # log to pl for checkpointing
        self.log('Val loss', loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        
        inputs = batch[0]
        targets = batch[1]
        targets = torch.flatten(targets)

        predictions = self(inputs)

        loss = self.loss_fn_base(predictions, targets)
        
        # log to mlflow
        self.logger.log_metrics({"Test loss" : loss}, step=self.global_step)
        # log to pl for checkpointing
        self.log('Test loss', loss)

        return loss
    
    def validation_epoch_end(self, valdation_step_outputs):
        
        loss_mean = torch.stack(valdation_step_outputs).mean()
        self.log("val_loss_mean", loss_mean)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.lr)
        
        return optim
