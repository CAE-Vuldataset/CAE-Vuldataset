import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, inputs_embeds=None, input_ids=None, position_idx=None, attn_mask=None, labels=None): 
        bs,l=input_ids.size()

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        if input_ids is not None:
            inputs_embeds=self.encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = inputs_embeds
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeds)
        inputs_embeds=inputs_embeds*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeds, attention_mask=attn_mask, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long())[0]
        logits=self.classifier(outputs)
        # shape: [batch_size, num_classes]
        prob=F.softmax(logits, dim=-1)
        input_size = input_ids.size() if input_ids is not None else inputs_embeds.size()
        assert prob.size(0) == input_size[0], (prob.size(), input_size)
        assert prob.size(1) == 2, prob.size()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
    
    def get_representation(self, input_ids=None, position_idx=None, attn_mask=None):
        bs,l=input_ids.size()

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        if input_ids is not None:
            inputs_embeds=self.encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = inputs_embeds
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeds)
        inputs_embeds=inputs_embeds*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeds, attention_mask=attn_mask, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long(),output_hidden_states=True)[1]
        return outputs 


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

        
    def forward(self, hidden, inputs_embeds=None, input_ids=None, position_idx=None,
            attn_mask=None, labels=None, representation=False
        ): 
        hidden = hidden[12][ :, 0, :]

        bs,l=input_ids.size()
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        if input_ids is not None:
            inputs_embeds=self.encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = inputs_embeds
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeds)
        inputs_embeds=inputs_embeds*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeds, attention_mask=attn_mask, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long())
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
            if representation:
                return loss, prob, rp
            else:
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
        self.relu1 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  512)
        # self.dropout_4 = nn.Dropout(0.5)

        self.dense_3 = nn.Linear(1024,  config.hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss =  nn.CrossEntropyLoss()

        
    def forward(self, hidden, inputs_embeds=None, input_ids=None, position_idx=None,
            attn_mask=None, labels=None, representation=False
        ):
        hidden = hidden[12][ :, 0, :]

        bs,l=input_ids.size()
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        if input_ids is not None:
            inputs_embeds=self.encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = inputs_embeds
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeds)
        inputs_embeds=inputs_embeds*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeds, attention_mask=attn_mask, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long(), output_hidden_states=True)
        xp_ = outputs[1][self.args.early_layer][:, 0, :]


        xp_ = self.dense_2(xp_)
        xp_ = self.relu1(xp_)
        xp_ = self.dropout_2(xp_)

        # hidden = self.dense_4(hidden)
        # hidden = self.relu(hidden)
        # hidden = self.dropout_4(hidden)
        # print(hidden.shape, xp_.shape)

        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = self.relu2(x)
        x = self.dropout_3(x)
        rp = x.clone().cpu().detach().numpy()

        logits = self.out_proj(x)
        prob = self.softmax(logits)
        if labels is not None:
            loss = self.loss(logits, labels)
            if representation:
                return loss, prob, rp
            else:
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
        # self.relu1 = nn.ReLU()
        # self.dropout_2 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  512)
        # self.dropout_4 = nn.Dropout(0.5)

        self.dense_3 = nn.Linear(config.hidden_size * 2,  config.hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.5)

        # self.dense_4 = nn.Linear(config.hidden_size,  config.hidden_size)
        # self.dropout_4 = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.loss =  nn.CrossEntropyLoss()

        
    def forward(self, hidden, inputs_embeds=None, input_ids=None, position_idx=None,
            attn_mask=None, labels=None, representation=False
        ):
        hidden = hidden[12][ :, 0, :]

        bs,l=input_ids.size()
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        if input_ids is not None:
            inputs_embeds=self.encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = inputs_embeds
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeds)
        inputs_embeds=inputs_embeds*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeds, attention_mask=attn_mask, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long(), output_hidden_states=True)
        xp_ = outputs[1][4][:, 0, :]

        # xp_ = self.dense_2(xp_)
        # xp_ = self.relu1(xp_)
        # xp_ = self.dropout_2(xp_)

        # hidden = self.dense_4(hidden)
        # hidden = self.relu(hidden)
        # hidden = self.dropout_4(hidden)
        x = torch.cat((hidden, xp_), dim=1)
        x = self.dense_3(x)
        x = self.relu2(x)
        x = self.dropout_3(x)
        rp = x.clone().cpu().detach().numpy()

        logits = self.out_proj(x)
        prob = self.softmax(logits)
        if labels is not None:
            loss = self.loss(logits, labels)
            if representation:
                return loss, prob, rp
            else:
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
    
    def embed(self,  input_ids=None, position_idx=None, attn_mask=None ):
        bs,l=input_ids.size()

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        if input_ids is not None:
            inputs_embeds=self.model.encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = inputs_embeds
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeds)
        inputs_embeds=inputs_embeds*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        return inputs_embeds

    def forward(self, inputs_embeds=None,  position_idx=None, attn_mask=None, return_single_prob=True):
        outputs = self.model.encoder.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long()
        )[0]
        outputs = self.model.classifier(outputs)
        prob = F.softmax(outputs, dim=-1)
        # print(prob)
        if return_single_prob:
            return prob[:,prob.size(1) - 1].unsqueeze(1)
        return prob



class ModelClassifierWreapper(nn.Module):   
    def __init__(self, model, classifier, args):
        super(ModelClassifierWreapper, self).__init__()
        self.model = model
        self.classifier = classifier
        self.args=args
    
    def embed(self,  input_ids=None, position_idx=None, attn_mask=None ):
        bs,l=input_ids.size()

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        if input_ids is not None:
            inputs_embeds=self.model.encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            inputs_embeds = inputs_embeds
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeds)
        inputs_embeds=inputs_embeds*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        return inputs_embeds

    def forward(self, input_embeds, xpos=None, xatt=None, xpinput=None, xppos=None, xpatt=None,  return_single_prob=True):
        representation = self.model.encoder.roberta(
            inputs_embeds=input_embeds,
            attention_mask=xatt, position_ids=xpos, token_type_ids=xpos.eq(-1).long(),
            output_hidden_states=True
        )[1]
        # print(representation.shape)
        classifier_output = self.classifier(representation, input_ids=xpinput, position_idx=xppos, attn_mask=xpatt)

        if return_single_prob:
            return classifier_output[:, classifier_output.size(1) - 1].unsqueeze(1)
        return classifier_output
