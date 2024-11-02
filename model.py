import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import parameters as param

# Defining the model class

class CustomLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim, target_dim, n_layers, vocabulary_size):
        """
        :param input_dim: int. The number of inputs for the LSTM network
        :param hidden_dim: int. The dimension of the LSTM network
        :param embedding_dim: int. The dimension of the embedding layer
        :param target_dim: int. The dimension of the output
        :param n_layers: int; The number of stacked LSTM cells
        :param vocabulary_size: int. The number of unique channels
        """
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embed_channel = nn.Embedding(vocabulary_size['channel_init']+1,
                                          embedding_dim,
                                          padding_idx=param.padding_value)
        self.embed_env_first = nn.Embedding(vocabulary_size['env_first_url'] + 1,
                                            embedding_dim,
                                            padding_idx=param.padding_value)
        self.embed_env_last = nn.Embedding(vocabulary_size['env_last_url'] + 1,
                                           embedding_dim,
                                           padding_idx=param.padding_value)
        self.lstm = nn.LSTM(len(param.features_to_encode) * embedding_dim +
                            input_dim,
                            self.hidden_dim, num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=param.model_param['p_dropout'])
        # The final output of the lstm (the last hidden state) is passed to a linear layer
        self.linear = nn.Linear(self.hidden_dim, target_dim)
        # The result of the linear layer is normalize through a softmax
        # (actually using the log of a softmax, no real reason to do this, but apparently pytorch has better
        # performances using the LogSoftmax and the NLLLoss rather than the standard function)
        self.output = nn.LogSoftmax(dim=1)

    def masked_log_softmax(self, input, mask=None):
        """
        :param input: torch.Tensor. The tensor for which we need to evaluate the log_softmax
        :param mask: torch.Tensor. The tensor with the indices on which we need to evaluate the log_softmax
        :return: output: torch.Tensor. The tensor with the log_softmax of the input
        """
        if mask is None:
            output = nn.functional.log_softmax(input, dim=1)
        else:
            output = torch.zeros(list(input.size()))
            for i in range(mask.size()[0]):
                output[i, :mask[i], :] = nn.functional.log_softmax(input[i, :mask[i], :], dim=0)
        return output

    def masked_exp(self, input, mask=None):
        """
        :param input: torch.Tensor. The tensor for which we need to evaluate the exponential
        :param mask: torch.Tensor. The tensor with the indices on which we need to evaluate the exponential
        :return: output: torch.Tensor. The tensor with the exponential of the input
        """
        if mask is None:
            output = torch.exp(input)
        else:
            output = torch.zeros(list(input.size()))
            for i in range(mask.size()[0]):
                output[i, :mask[i], :] = torch.exp(input[i, :mask[i], :])
        return output

    def attention(self, s, h):
        """
        :param s: pytorch tensor. The full output of the LSTM
        :param h: pytorch tensor. The last output of the LSTM
        :return: pytorch tensor. The attention weights
        """
        full_output, length = pad_packed_sequence(s, batch_first=True)
        last_output = h.view(-1, self.hidden_dim, param.model_param['layers'])
        dot_product = torch.matmul(full_output, last_output)
        attention_weights = self.masked_exp(self.masked_log_softmax(dot_product, length), length)
        return attention_weights

    def forward(self, x, l):
        """
        :param x: pytorch tensor. An array with the ordered channels in the path
        :param l: int. The real length of the path (or the admitted length)
        :return: scores: pytorch tensor. The probability of belonging to the output classes (if with attention, also the
        attention weights)
        """
        # Since now there are more than one feature we need to pass to the embedding vector only the appropriate one.
        # In this case so, we take the first element of the tensor (the encoded channel) and pass it to the embedding
        # layer. The output will be then concatenated together with the second feature and then passed on as in the
        # one feature case
        embedded_channel = self.embed_channel(x[:, :, 0].long())
        embedded_first_env = self.embed_env_first(x[:, :, 1].long())
        embedded_last_env = self.embed_env_last(x[:, :, 2].long())
        embedded_vector = torch.cat((embedded_channel, embedded_first_env, embedded_last_env),
                                    dim=2)
        # embedded_vector = self.dropout(embedded_vector)
        concat_input = torch.cat((embedded_vector, x[:, :, 3:].float().view(x[:, :, 3:].size()[0],
                                                                            x[:, :, 3:].size()[1],
                                                                            x.size()[2] -
                                                                            len(param.features_to_encode))),
                                 dim=2)
        pack_batch = pack_padded_sequence(concat_input,
                                          l,
                                          batch_first=True,
                                          enforce_sorted=False)
        out, (hn, cn) = self.lstm(pack_batch)
        weights = self.linear(hn[-1].view(-1, self.hidden_dim))
        scores = self.output(weights)

        # Running the attention function only in the evaluation phase
        if param.attribution_model == 'attention' and not self.training:
            att = self.attention(out, hn)
            return scores, att
        return scores
