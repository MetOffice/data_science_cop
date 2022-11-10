import pytorch_lightning as pl
import torch

# define MLP
class CloudBaseMLP(pl.LightningModule):
    def __init__(self, input_size, ff_nodes, output_size, layer_num, activation, lr=2e-3):
        super().__init__()
        
        if activation == "relu" :
            self.activation = torch.nn.ReLU
        elif activation == "tanh":
            self.activation = torch.nn.Tanh
        self.ff_nodes = ff_nodes
        self.input_size = input_size
        self.output_size = output_size
        self.layer_norm = torch.nn.LayerNorm((input_size))
        
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
        self.linears.append(torch.nn.Linear(ff_nodes[-1], self.output_size))
        
        # pytorch insists softmax normalization should be done outside of model forwards, so a function is defined in the model for this purpose
        self.normalize_outputs = torch.nn.Softmax()

        self.crossentropy_loss = torch.nn.CrossEntropyLoss()

        self.lr = lr

        self.save_hyperparameters()  # save hyperparameters for model checkpointing

    def forward(self, x):
        
        # layernorm
        x = self.layer_norm(x)
        
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
