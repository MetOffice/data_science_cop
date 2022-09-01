import pytorch_lightning as pl
import torch


# define MLP
class CloudBaseMLP(pl.LightningModule):
    def __init__(self, input_size, ff_nodes, output_size, lr=2e-3):
        super().__init__()

        self.input_size = input_size
        self.ff_nodes = ff_nodes
        self.output_size = output_size

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
        
        # flatten the per height level feats to be a single sample of feats

        out = self.sequential_layers(x)  # apply the sequential layers to the input

        final_prediction = out  # do no more with the output

        return final_prediction

    def normalize_outs(self, predictions):

        return self.normalize_outputs(predictions)

    def generic_model_step(self, batch, batch_idx, str_of_step_name):

        inputs = batch['x']
        targets = batch['cloud_base_target']
        
        # print(inputs.shape)
        inputs = torch.flatten(inputs, start_dim=1)

        predictions = self(inputs)

        loss = self.crossentropy_loss(predictions, targets)

        # log to tensorboard
        self.log((str_of_step_name + " loss"), loss)

        return loss

    def training_step(self, batch, batch_idx):

        return self.generic_model_step(batch, batch_idx, "training")

    def validation_step(self, batch, batch_idx):

        return self.generic_model_step(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):

        return self.generic_model_step(batch, batch_idx, "test")

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), self.lr)

        return optim
