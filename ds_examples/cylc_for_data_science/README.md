# Cylc for Data Science exmaples

This folder contain a number of small toy workflows that demonstrates different ways that cylc can be used for data science. This is particularly when doing these tasks on Met Office or other platforms where cylc is installed, to make it easier to work with those platforms. These include dummy tasks, where the propertask would be included, but include all the ley cylc features that would be needed.

Examples include
* `inference_workflow` - A simple demomstration of a workflow for running an ML weather model that looks similar to what is done for traditional physics-based NWP models.
* ` inference_ensemble` - The same as the inference workflow, but now showing how this could be done for an ensemble run.
* `data_transfer` - This demostrates how to run a regularly scheduled task, such as transferring data, much as one could do with a cron job, but through a cylc workflow.
* `hyperparameter_tuning` - This demonstrates a more complicated use of cylc, to orchestrate a grid search approach to hyperparameter tuning, and also breaking your training run into multiple tasks by epoch.