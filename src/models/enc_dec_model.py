import torch
import torch.nn as nn




class BertGpt2EncoderDecoder(nn.Module):  
    def __init__(self, bert_config, gpt2_config):  
        super().__init__()  
        self.encoder = BertModel(bert_config)  
        self.decoder = GPT2Model(gpt2_config)  
        self.classifier = nn.Linear(gpt2_config.n_embd, num_ner_labels)  
        
        # 连接适配层  
        self.adapter = nn.Linear(bert_config.hidden_size, gpt2_config.n_embd)  
        
    def forward(self, encoder_input, decoder_input, labels=None):  
        encoder_outputs = self.encoder(**encoder_input)  
        hidden_states = self.adapter(encoder_outputs.last_hidden_state)  
        
        decoder_outputs = self.decoder(  
            input_ids=decoder_input['input_ids'],  
            attention_mask=decoder_input['attention_mask'],  
            encoder_hidden_states=hidden_states  
        )  
        logits = self.classifier(decoder_outputs.last_hidden_state)  
        # 计算序列标注损失...  
        return logits  