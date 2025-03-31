import torch
import re
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from src.models.crf import CRF

from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import json

# from loader import load_vocab, encode_sentence
from typing import List, Dict, Union, Tuple
from src.models.bert.configuration_bert import BertConfig
from src.configs.config import NerConfig


from src.configs.config import BERT_MODEL_PATH


class ModelHub:
	'''
		choose your model to train
	'''
	def __init__(self, model_name, config):
		if model_name == "bert":
			self.model = BertCRFModel(config)
		elif model_name == "lstm":
			self.model = TorchModel(config)
		elif model_name=='sentence':
			self.model = WholeSentenceNERModel(config)
		else:
			raise NotImplementedError("model name not supported")



class BaseModel(nn.Module):
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else {
			"bert_model_name": BERT_MODEL_PATH,
			"max_seq_length": 512,
			"batch_size": 32,
			"learning_rate": 3e-4,
			"max_steps": 500,
			"output_dir": "./output",
			"lora_rank": 8,
			"ner_data_type": "chinese_ner_sft",
			'num_train_epochs': 2,
   			"hidden_size":768,
      		"num_layers":2,
      		"use_crf": True,
		}


class TorchModel(BaseModel):
	def __init__(self, config=None, ner_config:NerConfig = None):	
		super().__init__(config)
		hidden_size = self.config["hidden_size"]
		# 必须先跑loader，获得vocab_size
		max_length = self.config["max_seq_length"]
		class_num = ner_config.num_ner_labels
		num_layers = self.config["num_layers"]
  
		self.tokenizer = BertTokenizer(BERT_MODEL_PATH)
		self.vocab_size = self.tokenizer.vocab_size
		self.embedding = nn.Embedding(self.vocab_size, hidden_size, padding_idx=0)
		self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
		self.classify = nn.Linear(hidden_size * 2, class_num) 
		# crf层, 用来计算 emission score tensor
		self.crf_layer = CRF(class_num, batch_first=True)
		self.use_crf = self.config["use_crf"]
		# -1 is the padding value for labels, which will be ignored in loss calculation
		self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

	def forward(self, input_ids, attention_mask = None, labels = None):
		'''
			loss: (batch_size * seq_len, 1)
  		'''
		x = self.embedding(input_ids) # (batch_size, seq_len)
  
		x,_ = self.bilstm(x) # (batch_size, seq_len, hidden_size * 2)

		predict = self.classify(x) # (batch_size, seq_len, class_num)
		
		if labels is not None:
			if self.use_crf:
				mask = labels.gt(-1)
				# crf自带cross entropy loss
				# CRF loss 最后需要取反
				return - self.crf_layer(predict, labels, mask, reduction = 'mean')
		 		
			else:
				return self.loss(predict.view(-1, predict.shape[-1]), labels.view(-1))
		else:
			if self.use_crf:
				# 维特比解码 viterbi
				return self.crf_layer.decode(predict) # (batch_size, seq_len)
			else:
				return predict

class BertCRFModel():
	'''
		基于BERT的CRF模型
	'''
	def __init__(self, bert_config:BertConfig, ner_config:NerConfig):
		super().__init__()
		self.config = {
			"bert_model_path": BERT_MODEL_PATH,
			"class_num": ner_config.num_ner_labels,
			"hidden_size": bert_config.hidden_size,
			"dropout": bert_config.hidden_dropout_prob
		}
  
		self.bert = BertModel.from_pretrained(self.config["bert_model_path"], return_dict = False)
		self.classifier = nn.Linear(self.config["hidden_size"], self.config["class_num"])
		self.crf = CRF(self.config['class_num'], batch_first=True)
	def forward(self, input_ids, attention_mask = None, labels = None):
		sequence_output, _ = self.bert(input_ids) # (batch_size, seq_len, hidden_size)
		# print("sequence_output = \n", sequence_output)
		
		predicts = self.classifier(sequence_output) # (batch_size, seq_len, class_num)

		if labels!=None: # 计算CRF Loss
			mask = labels.gt(-1)
			loss = self.crf(predicts, labels, mask, reduction='mean') # (batch_size, seq_len, class_num)
			return -loss
		else:
			return self.crf.decode(predicts) # (batch_size, seq_len)
	
  
class WholeSentenceNERModel(nn.Module):
	'''
	  do the NER task for the entire sentence (sentence classification)
	'''
	def __init__(self, bert_config:BertConfig, ner_config:NerConfig, recurrent_type = "gru"):
		super().__init__()
		self.bert = BertModel.from_pretrained(BERT_MODEL_PATH, return_dict = False)
		self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
		self.num_labels = ner_config.num_ner_labels
		
		if recurrent_type == "lstm":
			self.recurrent_layer = nn.LSTM(bert_config.hidden_size, 
                                  bert_config.hidden_size//2, 
                                  batch_first=True, 
                                  bidirectional=True,
                                  num_layers = 1)
		elif recurrent_type == 'gru':
			self.recurrent_layer = nn.GRU(bert_config.hidden_size, 
                                 			bert_config.hidden_size//2,
											batch_first=True,
											bidirectional=True,
											num_layers =1
											)
		else:
			assert False
   
		
		self.classifier = nn.Linear(bert_config.hidden_size, self.num_labels)
		
			
	
	def forward(self, input_ids=None, attention_mask=None, labels=None):
		'''
			input_ids: (batch_size, seq_len)
			attention_mask: (batch_size, seq_len)
			labels: (batch_size, seq_len)
  		'''
    
		output = self.bert(input_ids, attention_mask) # (batch_size, seq_len, hidden_size)

		pooled_output = output[1] # (batch_size, hidden_size)
  
		pooled_output = self.dropout(pooled_output)
  
		recurrent_output,_ = self.recurrent_layer(pooled_output.unsqueeze(0)) # (1, batch_size, hidden_size) 
		
		# 线性层只能处理二维张量
		output = self.classifier(recurrent_output.squeeze(0)) # (batch_size, num_labels)
  
  
		if labels is not None:
			loss = nn.CrossEntropyLoss()
			return loss(output, labels.view(-1))
		else:
			return output
  



def choose_optimizer(config, model):
	optimizer = config['optimizer']
	learning_rate = config['learning_rate']
 
	if optimizer == 'adam':
		return Adam(model.parameters(), lr=learning_rate)
	elif optimizer == 'sgd':
		return SGD(model.parameters(), lr=learning_rate)
	elif optimizer == 'adamw':
		return AdamW(model.parameters(), lr=learning_rate)



def id_to_label(id, config):
	'''
	return label
	'''
	label2id = {}
	with open(config['schema_path'], 'r', encoding = 'utf8') as f:
		label2id = json.load(f)

	for k, v in label2id.items():
		if v == id:
			return k

		



if __name__ == '__main__':

 


	input = torch.LongTensor([input])
 
	print("input = \n",input)
 
	# output = model(input)
 
	# print(output)	
	# model = BertCRFModel(Config)
	# output = model(input)
	# print(output)	
	
 
	input_ids = torch.LongTensor([[1,3,34,67,64,678,123],[123,356,347,673,642,634,183]])
	attention_mask = torch.LongTensor([[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]])
	labels = torch.LongTensor([[1], [0]])
	model = WholeSentenceNERModel()
	output = model(input_ids, attention_mask, labels)
	print(output)
  