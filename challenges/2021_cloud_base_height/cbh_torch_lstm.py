import torch
import pytorch_lightning as pl


# define RNN
class CloudBaseLSTM(pl.LightningModule):
    def __init__(self, input_size, lstm_layers, lstm_hidden_size, output_size, height_dimension, embed_size, BILSTM=False, backward_lstm_differing_transitions=False, batch_first=True, lr=2e-3, skip_connection=False, norm_method = None, norm_mat_mean = None, norm_mat_std = None):
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
        self.norm_method = norm_method
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
            self.scale_input_layer = torch.nn.Conv1d(input_size+embed_size, output_size, 1)
        self.linearCap = torch.nn.Linear(output_size*height_dimension, height_dimension)
            
        self.save_hyperparameters() # save hyperparameters for 
        
        if self.norm_method == 'p_l_p_f' or self.norm_method == 'p_f':
            self.norm_mat_mean = norm_mat_mean
            self.norm_mat_std = norm_mat_std
        
        
    def forward(self, x):
        
        if self.norm_method == 'p_l_p_f' or self.norm_method == 'p_f':
            x = torch.subtract(x, self.norm_mat_mean)
            x = torch.div(x, self.norm_mat_std)
        elif self.norm_method == 'layer_relative':
            norm_mean = torch.mean(x, axis = 1, keepdims=True)
            nord_std = torch.std(x, axis = 1, keepdims=True)
            x = torch.sub(x, norm_mean)
            x = torch.div(x, nord_std)
        else:
            print("Error norm")
            raise Exception()
            
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
            # print(x_and_height.size())
            resid = torch.swapaxes(x_and_height,1,2)
            # print(resid.size())
            resid = self.scale_input_layer(resid)
            # print(resid.size())
            resid = torch.squeeze(resid)
            # print(resid.size())
            # print(lstm_out.size())
            lstm_out = resid + lstm_out
            
        
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
        self.logger.log_metrics({"val_loss_mean" : loss_mean}, step=self.global_step)
        self.log("val_loss_mean", loss_mean)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.lr)
        
        return optim
