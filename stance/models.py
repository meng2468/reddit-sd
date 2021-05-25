import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleStDClassifier(nn.Module):
  """ 
  SimpleStDClassifier
  --------------------------------------------------
  SimpleStDClassifier is meant to be trained on ONE specific target.

  Parameters:
    - base_model              : (nn.Module) base model, e.g. BERT, GPT...
    - output_dim              : (int) output dimension
    - base_model_output_size  : (int) base_model output dimension, e.g. bert=768
  
  Notes:
    - 
  """
  def __init__(self, base_model, output_dim, base_model_output_size=768):
    super().__init__()
    self.base_model = base_model
    self.output_dim = output_dim

    self.classifier = nn.Sequential(
        nn.Linear(base_model_output_size, output_dim)
    )

    # initialize weights
    for layer in self.classifier:
      if isinstance(layer, nn.Linear):
          layer.weight.data.normal_(mean=0.0, std=0.02)
          if layer.bias is not None:
              layer.bias.data.zero_()
    
  def forward(self, inputs, **args):
    # through language model
    hidden, pooler = self.base_model(inputs, return_dict=False)

    # sentence classification: ignore all but the first vector
    hidden = hidden[:,0,:]

    # through classifier
    logits = self.classifier(hidden)
    return logits

class StDClassifierQAVersion(nn.Module):
  """ 
  StDClassifierQAVersion
  --------------------------------------------------
  StDClassifier Question-Answering version with concatenation of input & targets within the model

  Parameters:
    - base_model              : (nn.Module) base model, e.g. BERT, GPT...
    - output_dim              : (int) output dimension
    - base_model_output_size  : (int) base_model output dimension, e.g. bert=768
    - dropout:                : (float) dropout rate
  
  Notes:
    - Makes more sense when defining Stance Detection as a Question-Answering problem, i.e. with multiple stances per dataset
      (which we don't have in the Reddit dataset)
  """
  def __init__(self, base_model, output_dim, base_model_output_size=768, dropout=0.5):
    super().__init__()
    self.base_model = base_model
    self.output_dim = output_dim

    self.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(base_model_output_size*2, base_model_output_size),
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
    assert isinstance(inputs, tuple), "Invalid input, must be a tuple of (inputs, targets)."
    inputs, targets = inputs

    # through language model
    hidden_inputs, pooler = self.base_model(inputs, return_dict=False)
    hidden_targets, _ = self.base_model(targets, return_dict=False)

    # sentence classification: ignore all but the first vector
    hidden_inputs = hidden_inputs[:,0,:].squeeze(1)
    hidden_targets = hidden_targets[:,0,:].squeeze(1)

    # concatenate
    hidden = torch.cat([hidden_inputs, hidden_targets], dim=1)    

    # through classifier
    logits = self.classifier(hidden)
    return logits


class StDClassifierWithTargetSpecificHeads(nn.Module):
  """ 
  StDClassifierWithTargetSpecificHeads 
  --------------------------------------------------
  StDClassifier with multiple classifiers on top of base_model (denoted as 'heads'), each playing for a different target.

  Parameters:
    - base_model              : (nn.Module) base model, e.g. BERT, GPT...
    - output_dim              : (int) output dimension
    - heads                   : (int) number of heads, i.e. number of targets in the dataset
    - base_model_output_size  : (int) base_model output dimension, e.g. bert=768
    - dropout:                : (float) dropout rate
  
  Notes:
    - Highly memory inefficient. It creates a "big" linear layer of size (plm_out_dim, plm_out_dim)*heads. 
      Thus, for e.g. the ARC dataset with 186 heads, the dimensions are (142,848, 142,848), i.e. 20+ billions parameters
    - Yields good performance on SemEval2016Task6, with 0.761 macro-F1-score over all 5 targets in test data.
    - Can think of this architecture as training one model per target. At least it is GPU optimized.
  
  References: 
    -
  """
  def __init__(self, base_model, output_dim, heads=1, base_model_output_size=768, dropout=0.5):
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
    assert isinstance(inputs, tuple), "Invalid input, must be a tuple of (inputs, targets)."
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
