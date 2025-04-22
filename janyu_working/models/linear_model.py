
# models/linear_model.py
class LinearModel(BaseModel):
     def __init__(self, config):
        super().__init__(config)
        # Define linear layers (possibly after flattening time steps)
        ...
     def forward(self, x): ...