import torch
import torch.nn as nn

class StDClassifier(nn.Module):
  def __init__(self, base_model, output_dim, base_model_output_size=768, dropout=.05):
    super().__init__()
    self.base_model = base_model
    self.output_dim = output_dim

    self.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(base_model_output_size, base_model_output_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(base_model_output_size, output_dim)
    )

    # initialize weights
    for layer in self.classifier:
      if isinstance(layer, nn.Linear):
          layer.weight.data.normal_(mean=0.0, std=0.02)
          if layer.bias is not None:
              layer.bias.data.zero_()
  
  def forward(self, inputs, **args):
    if isinstance(inputs, tuple): # UNFORTUNATELY, cannot use pack_padded_sequence with BERT...
      inputs, text_lengths = inputs
      inputs = nn.utils.rnn.pack_padded_sequence(inputs, text_lengths.to('cpu'))
      inputs = inputs.data
    hidden, pooler = self.base_model(inputs, return_dict=False)
    logits = self.classifier(hidden[:,0,:]) # sentence classification: ignore all but the first vector
    return logits
