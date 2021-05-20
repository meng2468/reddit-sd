import torch
import torch.nn as nn
import torch.nn.functional as F

class StDClassifier(nn.Module):
  def __init__(self, base_model, output_dim, heads=1, base_model_output_size=768, dropout=.05):
    super().__init__()
    self.base_model = base_model
    self.output_dim = output_dim
    self.heads = heads

    self.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(base_model_output_size*heads, base_model_output_size*heads),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(base_model_output_size*heads, output_dim)
    )

    # initialize weights
    for layer in self.classifier:
      if isinstance(layer, nn.Linear):
          layer.weight.data.normal_(mean=0.0, std=0.02)
          if layer.bias is not None:
              layer.bias.data.zero_()
    
  def forward(self, inputs, **args):
    # UNFORTUNATELY, cannot use pack_padded_sequence with BERT...
    # if isinstance(inputs, tuple): 
    #   inputs, text_lengths = inputs
    #   inputs = nn.utils.rnn.pack_padded_sequence(inputs, text_lengths.to('cpu'))
    #   inputs = inputs.data
    inputs, targets = inputs
    hidden, pooler = self.base_model(inputs, return_dict=False)
    hidden = hidden[:,0,:].squeeze(1) # sentence classification: ignore all but the first vector

    # modify hidden with mask
    mask = F.one_hot(targets, num_classes=self.heads) # (bs, n_targets) e.g. (32, 6)
    mask = torch.repeat_interleave(mask, hidden.shape[1], dim=1) # (bs, n_targest*outdim) e.g. (32, 4608)
    mask = mask.float()

    hidden = hidden.repeat(1, self.heads) * mask    
    hidden = hidden.reshape((inputs.shape[0], -1)) # reshape

    logits = self.classifier(hidden)
    return logits
