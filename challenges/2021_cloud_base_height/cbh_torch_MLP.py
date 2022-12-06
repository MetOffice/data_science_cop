import pytorch_lightning as pl
import torch

# define MLP
class CloudBaseMLP(pl.LightningModule):
    def __init__(self, input_size, ff_nodes, output_size, layer_num, activation, lr=2e-3, norm_method = None, norm_mat_mean = None, norm_mat_std = None):
        super().__init__()
        
        if activation == "relu" :
            self.activation = torch.nn.ReLU
        elif activation == "tanh":
            self.activation = torch.nn.Tanh
        self.ff_nodes = ff_nodes
        self.input_size = input_size
        self.output_size = output_size
        
        #init modellayers
        self.linears = torch.nn.ModuleList([])
        
        #input layer
        self.linears.append(torch.nn.Linear(self.input_size, ff_nodes[0]))
        self.linears.append(self.activation())
        #hidden layers
        for i in range(layer_num-1):
            self.linears.append(torch.nn.Linear(ff_nodes[i], ff_nodes[i+1]))
            self.linears.append(self.activation())
        #output layer
        self.linears.append(torch.nn.Linear(ff_nodes[layer_num-1], self.output_size))
        
        # pytorch insists softmax normalization should be done outside of model forwards, so a function is defined in the model for this purpose
        self.normalize_outputs = torch.nn.Softmax()

        self.crossentropy_loss = torch.nn.CrossEntropyLoss()
        self.norm_method = norm_method

        self.lr = lr

        self.save_hyperparameters()  # save hyperparameters for model checkpointing
        
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
        
        for layer in self.linears:
            x = layer(x)

        final_prediction = x  # do no more with the output

        return final_prediction

    def normalize_outs(self, predictions):

        return self.normalize_outputs(predictions)

    def training_step(self, batch, batch_idx):

        inputs = batch[0]
        targets = batch[1]
        
        # print(inputs.shape)
        inputs = torch.flatten(inputs, start_dim=1)

        predictions = self(inputs)

        loss = self.crossentropy_loss(predictions, targets)
        
        # log to mlflow
        self.logger.log_metrics({"Train loss" : loss}, step=self.global_step)
        # log to pl for checkpointing
        self.log('Train loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        inputs = batch[0]
        targets = batch[1]
        
        # print(inputs.shape)
        inputs = torch.flatten(inputs, start_dim=1)

        predictions = self(inputs)

        loss = self.crossentropy_loss(predictions, targets)

        # log to mlflow
        self.logger.log_metrics({"Val loss" : loss}, step=self.global_step)
        # log to pl for checkpointing
        self.log('Val loss', loss)

        return loss
    
    def validation_epoch_end(self, valdation_step_outputs):
        
        loss_mean = torch.stack(valdation_step_outputs).mean()
        self.logger.log_metrics({"val_loss_mean" : loss_mean}, step=self.global_step)
        self.log("val_loss_mean", loss_mean)


    def test_step(self, batch, batch_idx):

        inputs = batch[0]
        targets = batch[1]
        
        # print(inputs.shape)
        inputs = torch.flatten(inputs, start_dim=1)

        predictions = self(inputs)

        loss = self.crossentropy_loss(predictions, targets)

        # log
        self.logger.log_metrics({"Test loss" : loss}, step=self.global_step)
        # log to pl for checkpointing
        self.log('Test loss', loss)
        return loss

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), self.lr)

        return optim

    def on_train_end(self) -> None:
        """
        Called after training ends
        """
        self.logger.after_save_checkpoint()
