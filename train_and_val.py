import parameters as param
import torch
import numpy as np
import utilities as custom_util
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def train_and_val(dl_train, dl_val, rnn, optimizer, loss_function, output_name, rnn_parameters, rnn_parameters_name):
    """
    Train and validate the model
    :param dl_train: torch.dataloader. Dataloader for training data
    :param dl_val: torch.dataloader. Dataloader for validation data
    :param rnn: CustomLSTM. Model to train
    :param optimizer: torch.optim. Model optimizer
    :param loss_function: torch.loss. Loss function
    :param output_name: str, the name of the output file
    :param rnn_parameters: dict, the dictionary containing the parameters of the model
    :param rnn_parameters_name: str, the name of the file where to store the model parameters
    :return
    """
    # Initialising the output array with the training results
    output_train_val = np.zeros((param.model_param['epochs'], 6))
    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)
    for e in range(param.model_param['epochs']):
        training_loss = 0.
        # Looping to create training batches
        for x, y, l in dl_train:
            # Setting to zero the gradient of the model parameters (because pytorch accumulate the gradients)
            optimizer.zero_grad()
            # Running the model
            output = rnn(x, l)
            # Evaluating the loss function
            loss = loss_function(output, torch.LongTensor(y))
            # Running backward propagation for evaluating the gradient of the loss function
            loss.backward()
            # Optimizing the weights of the model
            optimizer.step()
            # Updating the training loss
            training_loss += loss.item()
        # Validation part. Setting to zero the validation loss and the accuracy
        test_loss = 0.
        accuracy = 0.
        # Initialising the confusion matrix
        c_matrix = np.zeros((2, 2))
        with torch.no_grad():
            # Setting the model in evaluation mode
            rnn.eval()
            # Looping through the validation batches
            for x, y, l in dl_val:
                # Running the model
                output = rnn(x, l)
                # Evaluating the validation loss function
                loss = loss_function(output, torch.LongTensor(y))
                test_loss += loss.item()
                # Evaluating the accuracy
                top_i, top_class = torch.exp(output).topk(1, dim=1)
                equals = top_class == torch.LongTensor(y).view(*top_class.shape)
                # accuracy += torch.mean(equals.type(torch.FloatTensor))
                accuracy += torch.sum(equals.type(torch.FloatTensor))
                c_matrix += custom_util.confusion_matrix(y.numpy(), top_class.numpy().reshape(-1, ))
        # Reinitialising the training mode for the model
        rnn.train()
        # Updating the learning rate
        # scheduler.step()
        # Logging the resulting loss function
        f1_score = 2*c_matrix[0, 0]/(np.sum(c_matrix, axis=1)[0]) *\
                   c_matrix[0, 0]/(np.sum(c_matrix, axis=0)[0]) /\
                   (c_matrix[0, 0]/(np.sum(c_matrix, axis=1)[0])+c_matrix[0, 0]/(np.sum(c_matrix, axis=0)[0]))
        logger.info('Epoch: ' + str(e) + '; Training loss: ' + str(training_loss / len(dl_train))
                    + '; Val loss: ' + str(test_loss / len(dl_val)) +
                    '; Accuracy: ' + str(accuracy.item() / len(dl_val)) + '; F1-Score: ' +
                    str(f1_score) +
                    '; Recall: ' + str(c_matrix[0][0]/(np.sum(c_matrix, axis=0)[0])) +
                    '; Negative recall: ' + str(c_matrix[1][1]/(np.sum(c_matrix, axis=0)[1])))
        # Preparing the output array
        output_train_val[e, :] = [training_loss / len(dl_train),
                                  test_loss / len(dl_val), accuracy.item() / len(dl_val),
                                  f1_score, c_matrix[0][0]/(np.sum(c_matrix, axis=0)[0]),
                                  c_matrix[1][1]/(np.sum(c_matrix, axis=0)[1])]
    # Saving the output array with the performances of the model
    df_output = pd.DataFrame(output_train_val, columns=['training_loss', 'validation_loss', 'accuracy',
                                                        'f1_score', 'recall', 'negative_recall'])
    df_output.to_csv(param.local_dir + output_name)
    # Saving the model parameters
    if param.dump_model:
        logger.info('Saving the model')
        logger.info('Loading the model parameters from storage')
        rnn_parameters['state_dict'] = rnn.state_dict()
        # checkpoint = {'input_size': param.model_param['input_dim'],
        #               'hidden_dimension': param.model_param['hidden_dim'],
        #               'vocabulary_size': param.model_param['voc_size'],
        #               'embedding_dimension': param.model_param['embedding_dim'],
        #               'output_size': param.model_param['target_dim'],
        #               'state_dict': rnn.state_dict()}
        # checkpoint_label = 'model_checkpoint_e' + str(param.model_param['epochs'])
        # checkpoint_name = 'model_checkpoint_hd_{hd}_ed_{ed}_nl_{nl}_epoch_{epoch}'.format(hd=str(
        #     param.model_param['hidden_dim']),
        #     ed=str(param.model_param['embedding_dim']),
        #     nl=str(param.model_param['layers']),
        #     epoch=str(param.model_param['epochs'])
        # )
        torch.save(rnn_parameters, param.local_dir + rnn_parameters_name)
    # Saving additional information for further analysis
    if param.deep_analysis:
        np.save(param.saving_folder + 'confusion_matrix.npy', c_matrix)
        np.save(param.saving_folder + 'weight.npy', rnn.linear.weight.detach().numpy())
        np.save(param.saving_folder + 'bias.npy', rnn.linear.bias.detach().numpy())
    return
