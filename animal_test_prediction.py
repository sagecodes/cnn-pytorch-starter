#%%
# imports
# Set paths for custom modules
sys.path.insert(0, './helpers')
sys.path.insert(0, './models')
from resnet50 import Resnet50_pretrained
from model_helpers import load_model

#%%
# Load model
# Define architecture
res_model = Resnet50_pretrained(3)
# Load weights, set ready for prediction
res_model = load_model(res_model, 'trained_models/test_train.pt',True)

# run prediction

# %%
