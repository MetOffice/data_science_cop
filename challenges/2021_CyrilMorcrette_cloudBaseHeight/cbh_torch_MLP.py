import pytorch_lightning as pl
import torch


# define MLP
class CloudBaseMLP(pl.LightningModule):
    def __init__(self, input_size, ff_nodes, output_size, lr=2e-3):
        super().__init__()
            
        self.input_size = input_size
        self.ff_nodes = ff_nodes
        self.output_size = output_size
        self.layer_norm = torch.nn.LayerNorm((input_size))
        self.sequential_layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.ff_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_nodes, self.ff_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_nodes, self.ff_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_nodes, self.ff_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_nodes, self.output_size),
        )  # pytorch insists softmax normalization should be done outside of model forwards, so a function is defined in the model for this purpose

        self.normalize_outputs = torch.nn.Softmax()

        self.crossentropy_loss = torch.nn.CrossEntropyLoss()

        self.lr = lr

        self.save_hyperparameters()  # save hyperparameters for model checkpointing

    def forward(self, x):
        
        # layernorm
        norm_x = self.layer_norm(x)
        
        out = self.sequential_layers(norm_x)  # apply the sequential layers to the input

        final_prediction = out  # do no more with the output

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
