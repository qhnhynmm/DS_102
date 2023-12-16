from model.rnn import createRNN_Model,RNN_Model
from model.lstm import createLSTM_Model, LSTM_Model
from model.gru import createGRU_Model, GRU_Model 

def build_model(config, answer_space):
    if config['model']['type_model']=='rnn':
        return createRNN_Model(config,answer_space)
    if config['model']['type_model']=='lstm':
        return createLSTM_Model(config,answer_space)
    if config['model']['type_model']=='gru':
        return createGRU_Model(config,answer_space)
    
def get_model(config, num_labels):
    if config['model']['type_model']=='rnn':
        return RNN_Model(config,num_labels)
    if config['model']['type_model']=='lstm':
        return LSTM_Model(config,num_labels)
    if config['model']['type_model']=='gru':
        return GRU_Model(config,num_labels)