from azure.ml.core import Workspace  
from azure.ml import MLClient  
  
# Get current workspace  
workspace = Workspace.from_config()  
  
# Create ML client with current workspace  
client = MLClient(workspace=workspace) 