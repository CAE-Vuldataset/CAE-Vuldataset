# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


class Model(nn.Module):   
    def __init__(self, encoder,config,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)

        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss, prob
        else:
            return prob
    
    def get_representation(self, input_ids=None):
        return self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)[1]
    
    def get_embedding(self, input_ids=None):
        return self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)[1][0][:, :, :]
 


class ClassifierA1(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config


        classifier_dropout = 0.5
        self.dropout_0 = nn.Dropout(classifier_dropout)

        self.lstm = nn.LSTM(config.hidden_size, 256, 3,  batch_first=True, dropout=0.7, bidirectional=True)


        # self.dense_2 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_2 = nn.Dropout(classifier_dropout)

        self.dense_3 = nn.Linear(config.hidden_size + 256*2,  config.hidden_size)
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

        self.loss =  nn.CrossEntropyLoss()
    

    def forward(self, hidden, xp, labels=None, representation=False):
        hidden = hidden[12][:, 0, :]
        xp_ = self.dropout_0(xp)

        output, (_, _) = self.lstm(xp_)
        xp_ = output[:, -1, :]

        # xp_ = self.dense_2(xp_)
        # xp_ = torch.tanh(xp_)
        # xp_ = self.dropout_2(xp_)


        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = torch.tanh(x)
        x = self.dropout_3(x)
        rp = x.clone().cpu().detach().numpy()

        # x = self.dense_4(x)
        # x = torch.tanh(x)
        # x = self.dropout_4(x)

        logits = self.out_proj(x)
        prob = self.softmax(logits)
        if labels is not None:
            loss = self.loss(logits, labels)
            # labels=labels.float()
            # loss=torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = -loss.mean()
            if representation:
                return loss, prob, rp
            return loss, prob
        else:
            if representation:
                return prob, rp
            return prob
 


class ClassifierA2(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config


        classifier_dropout = 0.5
        self.dropout_0 = nn.Dropout(classifier_dropout)

        self.lstm = nn.LSTM(config.hidden_size, 256, 3,  batch_first=True, dropout=0.7, bidirectional=True)


        # self.dense_2 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_2 = nn.Dropout(classifier_dropout)

        self.dense_3 = nn.Linear(config.hidden_size + 256*2,  config.hidden_size)
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

        self.loss =  nn.CrossEntropyLoss()
    

    def forward(self, hidden, xp, labels=None):
        hidden = torch.mean(hidden[12][:,:,:], 1)
        xp_ = self.dropout_0(xp)

        output, (_, _) = self.lstm(xp_)
        xp_ = output[:, -1, :]

        # xp_ = self.dense_2(xp_)
        # xp_ = torch.tanh(xp_)
        # xp_ = self.dropout_2(xp_)


        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = torch.tanh(x)
        x = self.dropout_3(x)

        # x = self.dense_4(x)
        # x = torch.tanh(x)
        # x = self.dropout_4(x)

        logits = self.out_proj(x)
        prob = self.softmax(logits)
        if labels is not None:
            loss = self.loss(logits, labels)
            # labels=labels.float()
            # loss=torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = -loss.mean()
            return loss, prob
        else:
            return prob
 

class ClassifierB1(nn.Module):   
    def __init__(self, encoder, config, args):
        super(ClassifierB1, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dense_2 = nn.Linear(config.hidden_size,  256)
        self.dropout_2 = nn.Dropout(0.5)

        self.dense_4 = nn.Linear(config.hidden_size,  512)
        self.dropout_4 = nn.Dropout(0.5)

        self.dense_3 = nn.Linear(768,  config.hidden_size)
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.loss =  nn.CrossEntropyLoss()

        
    def forward(self, hidden, input_ids=None,labels=None): 
        hidden = hidden[12][ :, 0, :]
        outputs=self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1))
        last_hidden_state = outputs.last_hidden_state

        xp_ = last_hidden_state[:, 0, :]
        xp_ = self.dense_2(xp_)
        xp_ = self.relu(xp_)
        xp_ = self.dropout_2(xp_)

        hidden = self.dense_4(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout_4(hidden)

        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = self.relu(x)
        x = self.dropout_3(x)

        logits = self.out_proj(x)
        prob = self.softmax(logits)

        if labels is not None:
            loss = self.loss(logits, labels)
            # labels=labels.float()
            # loss=torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = -loss.mean()
            return loss, prob
        else:
            return prob



class ClassifierB2(nn.Module):   
    def __init__(self, encoder, config, args):
        super(ClassifierB2, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dense_2 = nn.Linear(config.hidden_size,  256)
        self.dropout_2 = nn.Dropout(0.5)

        self.dense_4 = nn.Linear(config.hidden_size,  512)
        self.dropout_4 = nn.Dropout(0.5)

        self.dense_3 = nn.Linear(768,  config.hidden_size)
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.loss =  nn.CrossEntropyLoss()

        
    def forward(self, hidden, input_ids=None,labels=None):
        hidden = torch.mean(hidden[12][:, :, :], 1)
        outputs=self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1))
        last_hidden_state = outputs.last_hidden_state
        
        xp_ = last_hidden_state[:, 0, :]
        xp_ = self.dense_2(xp_)
        xp_ = self.relu(xp_)
        xp_ = self.dropout_2(xp_)

        hidden = self.dense_4(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout_4(hidden)

        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = self.relu(x)
        x = self.dropout_3(x)

        logits = self.out_proj(x)
        prob = self.softmax(logits)

        if labels is not None:
            loss = self.loss(logits, labels)
            # labels=labels.float()
            # loss=torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = -loss.mean()
            return loss, prob
        else:
            return prob



class ClassifierB3(nn.Module):   
    def __init__(self, encoder, config, args):
        super(ClassifierB3, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dense_2 = nn.Linear(config.hidden_size,  512)
        self.dropout_2 = nn.Dropout(0.5)

        self.dense_4 = nn.Linear(config.hidden_size,  512)
        self.dropout_4 = nn.Dropout(0.5)

        self.dense_3 = nn.Linear(1024,  config.hidden_size)
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.loss =  nn.CrossEntropyLoss()

        
    def forward(self, hidden, input_ids=None,labels=None):
        hidden = torch.mean(hidden[12][ :, :, :], 1)
        outputs=self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1))
        last_hidden_state = outputs.last_hidden_state
        
        xp_ = last_hidden_state[:, 0, :]
        xp_ = self.dense_2(xp_)
        xp_ = self.relu(xp_)
        xp_ = self.dropout_2(xp_)

        hidden = self.dense_4(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout_4(hidden)

        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = self.relu(x)
        x = self.dropout_3(x)

        logits = self.out_proj(x)
        prob = self.softmax(logits)

        if labels is not None:
            loss = self.loss(logits, labels)
            # labels=labels.float()
            # loss=torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = -loss.mean()
            return loss, prob
        else:
            return prob


class ClassifierB4(nn.Module):   
    def __init__(self, encoder, config, args):
        super(ClassifierB4, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dense_2 = nn.Linear(config.hidden_size,  512)
        self.relu1 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.5)

        self.dense_4 = nn.Linear(config.hidden_size,  512)
        self.relu2 = nn.ReLU()
        self.dropout_4 = nn.Dropout(0.5)

        self.dense_3 = nn.Linear(1024,  config.hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss =  nn.CrossEntropyLoss()

        
    def forward(self, hidden, input_ids=None,labels=None, representation=False): 
        hidden = hidden[12][ :, 0, :]
        outputs=self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1))
        last_hidden_state = outputs.last_hidden_state

        xp_ = last_hidden_state[:, 0, :]
        xp_ = self.dense_2(xp_)
        xp_ = self.relu1(xp_)
        xp_ = self.dropout_2(xp_)

        hidden = self.dense_4(hidden)
        hidden = self.relu2(hidden)
        hidden = self.dropout_4(hidden)

        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = self.relu3(x)
        x = self.dropout_3(x)
        rp = x.clone().cpu().detach().numpy()

        logits = self.out_proj(x)
        prob = self.softmax(logits)

        if labels is not None:
            loss = self.loss(logits, labels)
            # labels=labels.float()
            # loss=torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = -loss.mean()
            if representation:
                return loss, prob, rp
            return loss, prob
        else:
            if representation:
                return prob, rp
            return prob


class ClassifierB6(nn.Module):   
    def __init__(self, encoder, config, args):
        super(ClassifierB6, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args

        self.dense_2 = nn.Linear(config.hidden_size,  256)
        self.relu2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  512)
        # self.dropout_4 = nn.Dropout(0.5)

        self.dense_3 = nn.Linear(1024,  config.hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss =  nn.CrossEntropyLoss()

        
    def forward(self, hidden, input_ids=None, labels=None, representation=False): 
        hidden = hidden[12][ :, 0, :]
        outputs=self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)
        xp_ = outputs[1][self.args.early_layer][:, 0, :]

        xp_ = self.dense_2(xp_)
        xp_ = self.relu2(xp_)
        xp_ = self.dropout_2(xp_)

        # hidden = self.dense_4(hidden)
        # hidden = self.relu(hidden)
        # hidden = self.dropout_4(hidden)

        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = self.relu3(x)
        x = self.dropout_3(x)
        rp = x.clone().cpu().detach().numpy()

        logits = self.out_proj(x)
        prob = self.softmax(logits)

        if labels is not None:
            loss = self.loss(logits, labels)
            # labels=labels.float()
            # loss=torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = -loss.mean()
            if representation:
                return loss, prob, rp
            return loss, prob
        else:
            if representation:
                return prob, rp
            return prob
        

class ClassifierB7(nn.Module):   
    def __init__(self, encoder, config, args):
        super(ClassifierB7, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args

        # self.dense_2 = nn.Linear(config.hidden_size,  256)
        # self.dropout_2 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  512)
        # self.dropout_4 = nn.Dropout(0.5)

        self.dense_3 = nn.Linear(config.hidden_size * 2,  config.hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

        self.loss =  nn.CrossEntropyLoss()

        
    def forward(self, hidden, input_ids=None, labels=None, representation=False): 
        hidden = hidden[12][ :, 0, :]
        outputs=self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)
        xp_ = outputs[1][4][:, 0, :]

        # xp_ = self.dense_2(xp_)
        # xp_ = self.relu(xp_)
        # xp_ = self.dropout_2(xp_)

        # hidden = self.dense_4(hidden)
        # hidden = self.relu(hidden)
        # hidden = self.dropout_4(hidden)

        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = self.relu3(x)
        x = self.dropout_3(x)
        rp = x.clone().cpu().detach().numpy()

        logits = self.out_proj(x)
        prob = self.softmax(logits)

        if labels is not None:
            loss = self.loss(logits, labels)
            # labels=labels.float()
            # loss=torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = -loss.mean()
            if representation:
                return loss, prob, rp
            return loss, prob
        else:
            if representation:
                return prob, rp
            return prob

class ModelWreapper(nn.Module):   
    def __init__(self, model, args):
        super(ModelWreapper, self).__init__()
        self.model = model
        self.args=args
    
    def embed(self, input_ids):
        return self.model.encoder.roberta.embeddings.word_embeddings(input_ids)

    def forward(self, input_embeds, attention_mask=None, return_single_prob=True):
        outputs = self.model.encoder.roberta(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )[0]
        outputs = self.model.encoder.classifier(outputs)
        logits = self.model.dropout(outputs)
        prob = torch.sigmoid(logits)
        # print(prob)
        if return_single_prob:
            return prob[: ,prob.size(1) - 1].unsqueeze(1)
        return prob


class ModelClassifierWreapper(nn.Module):   
    def __init__(self, model, classifier, args):
        super(ModelClassifierWreapper, self).__init__()
        self.model = model
        self.classifier = classifier
        self.args=args
    
    def embed(self, input_ids):
        return self.model.encoder.roberta.embeddings.word_embeddings(input_ids)

    def forward(self, input_embeds, attention_mask=None, xp_ids=None, return_single_prob=True):
        representation = self.model.encoder.roberta(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )[1]
        classifier_output = self.classifier(representation, input_ids=xp_ids)

        if return_single_prob:
            return classifier_output[:, classifier_output.size(1) - 1].unsqueeze(1)
        return classifier_output
