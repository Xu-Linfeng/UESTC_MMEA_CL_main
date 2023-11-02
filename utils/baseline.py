import torch
from model import TBN
import copy

from torch import nn
from torch.nn.init import normal_, constant_
from context_gating import Context_Gating
from multimodal_gating import Multimodal_Gated_Unit
from ops.basic_ops import ConsensusModule

class Fusion_Network(nn.Module):

    def __init__(self, feature_dim, modality, midfusion, dropout):
        super().__init__()
        self.modality = modality
        self.midfusion = midfusion
        self.dropout = dropout

        if len(self.modality) > 1:  # Fusion

            if self.midfusion == 'concat':
                self._add_audiovisual_fc_layer(len(self.modality) * feature_dim, 512)

            elif self.midfusion == 'context_gating':
                self._add_audiovisual_fc_layer(len(self.modality) * feature_dim, 512)
                self.context_gating = Context_Gating(512)

            elif self.midfusion == 'multimodal_gating':
                self.multimodal_gated_unit = Multimodal_Gated_Unit(feature_dim, 512)
                if self.dropout > 0:
                    self.dropout_layer = nn.Dropout(p=self.dropout)

        else:  # Single modality
            if self.dropout > 0:
                self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, inputs):

        if len(self.modality) > 1:  # Fusion
            if self.midfusion == 'concat':
                base_out = torch.cat(inputs, dim=1)
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
            elif self.midfusion == 'context_gating':
                base_out = torch.cat(inputs, dim=1)
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
                base_out = self.context_gating(base_out)
            elif self.midfusion == 'multimodal_gating':
                base_out = self.multimodal_gated_unit(inputs)
        else:  # Single modality
            base_out = inputs[0]

        if self.dropout > 0:
            base_out = self.dropout_layer(base_out)
        
        output = {'features': base_out}

        return output

    def _add_audiovisual_fc_layer(self, input_dim, output_dim):

        self.fc1 = nn.Linear(input_dim, output_dim)
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        std = 0.001
        normal_(self.fc1.weight, 0, std)
        constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()

class Classification_Network(nn.Module):
    def __init__(self, feature_dim, modality, num_class,
                 consensus_type, before_softmax, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.reshape = True
        self.consensus = ConsensusModule(consensus_type)
        self.before_softmax = before_softmax
        self.num_segments = num_segments

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        if len(self.modality) > 1:  # Fusion
            self._add_classification_layer(512)
        else:  # Single modality
            self._add_classification_layer(feature_dim)

    def _add_classification_layer(self, input_dim):

        std = 0.001
        if isinstance(self.num_class, (list, tuple)):  # Multi-task

            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
            normal_(self.fc_verb.weight, 0, std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(input_dim, self.num_class)
            normal_(self.fc_action.weight, 0, std)
            constant_(self.fc_action.bias, 0)
            self.weight = self.fc_action.weight
            self.bias = self.fc_action.bias

    def forward(self, inputs):

        # Snippet-level predictions and temporal aggregation with consensus
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            base_out_verb = self.fc_verb(inputs)
            if not self.before_softmax:
                base_out_verb = self.softmax(base_out_verb)
            if self.reshape:
                base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
            output_verb = self.consensus(base_out_verb)

            # Noun
            base_out_noun = self.fc_noun(inputs)
            if not self.before_softmax:
                base_out_noun = self.softmax(base_out_noun)
            if self.reshape:
                base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])
            output_noun = self.consensus(base_out_noun)

            output = (output_verb.squeeze(1), output_noun.squeeze(1))

        else:
            base_out = self.fc_action(inputs)
            if not self.before_softmax:
                base_out = self.softmax(base_out)
            if self.reshape:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

            output = self.consensus(base_out)
            output = output.squeeze(1)

        return {'logits': output}


class Baseline(nn.Module):
    def __init__(self, num_segments, modality, base_model='BNInception',
                 new_length=None, consensus_type='avg', before_softmax=True,
                 dropout=0.8, midfusion='concat',):
        super().__init__()

        self.num_segments = num_segments
        self.modality = modality
        self.base_model = base_model
        self.new_length = new_length
        self.dropout = dropout
        self.before_softmax = before_softmax
        self.consensus_type = consensus_type
        self.midfusion = midfusion
        
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")
        
        self.feature_extract_network = TBN(self.num_segments, self.modality,
                                           self.base_model, self.new_length, 
                                           self.dropout)

        self.fusion_network = Fusion_Network(1024, self.modality, self.midfusion, self.dropout)

        self.feature_extractor = nn.Sequential(
            self.feature_extract_network,
            self.fusion_network
        )

        self.fc = None
        
        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.feature_extract_network.new_length, 
                   consensus_type, self.dropout)))

    @property
    def feature_dim(self):
        if len(self.modality) > 1:
            return 512
        else:
            return 1024

    def extract_vector(self, x):
        return self.feature_extractor(x)['features']

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.fc(x['features'])
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = Classification_Network(1024, self.modality, nb_classes, self.consensus_type, 
                                    self.before_softmax, self.num_segments)

        if self.fc is not None:
            nb_output = self.fc.num_class
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.fc_action.weight.data[:nb_output] = weight
            fc.fc_action.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
