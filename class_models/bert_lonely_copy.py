import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# from transformers import BertModel, BertLMHeadModel, BertConfig
from transformers import RobertaModel, RobertaTokenizer, RobertaForCausalLM, RobertaConfig

from class_models.loss import focal_loss
from class_models.bert import BertConfig, BertModel, BertLMHeadModel
from class_models.utils import load_checkpoint, tie_encoder_decoder_weights, set_trainable

class BertLonely(nn.Module):
    def __init__(self,
                 configs,
                 num_layers=1,
                 dropout_rate=0.5
                 ):
        super().__init__()

        # Configs
        self.configs = configs

        # Tokenizer
        if self.configs['type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.configs['bert_model'])
        elif self.configs['type'] == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.configs['bert_model'])
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]

        # Text Encoder
        if self.configs['type'] == 'bert':
            bert_config = BertConfig.from_json_file(self.configs['bert_config'])
            self.text_encoder = BertModel.from_pretrained(self.configs['pretrain_path'], config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)
        elif self.configs['type'] == 'roberta':
            self.text_encoder = RobertaModel.from_pretrained(self.configs['bert_model'], add_pooling_layer=False)

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        total = sum([param.nelement() for param in self.text_encoder.parameters()])
        print('Text Encoder Number of Params: %.2fM' % (total / 1e6))

        # Text Decoder
        if self.configs['type'] == 'bert':
            self.text_decoder = BertLMHeadModel.from_pretrained(self.configs['bert_model'], config=configs['bert_config'], ignore_mismatched_sizes=True)
        if self.configs['type'] == 'roberta':
            self.text_decoder = RobertaForCausalLM.from_pretrained('roberta-base')

        total = sum([param.nelement() for param in self.text_decoder.parameters()])
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        print('Text Decoder Number of Params: %.2fM' % (total / 1e6))

        # Tie Encoder and Decoder
        print("\n********** Tie Encoder and Decoder **********")
        if self.configs['type'] == 'bert':
            tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')
        elif self.configs['type'] == 'roberta':
            tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.roberta, '', '/attention')

        # Freeze parameters of the text_encoder and text_decoder to not train them
        if self.configs['type'] == 'bert':
            encoder_layers_to_train = ['encoder.layer.11.']
            decoder_layers_to_train = ['bert.encoder.layer.11.']
        elif self.configs['type'] == 'roberta':
            encoder_layers_to_train = ['encoder.layer.11.']
            decoder_layers_to_train = ['roberta.encoder.layer.11.']
        set_trainable(model_component=self.text_encoder, layer_names=encoder_layers_to_train, type='encoder', include_predictions=False, include_embeddings=False)
        set_trainable(model_component=self.text_decoder, layer_names=decoder_layers_to_train, type='decoder', include_predictions=True, include_embeddings=False)

        print("\n********** Text Encoder Trainable Status **********")
        for name, param in self.text_encoder.named_parameters():
            print(f"{name:60} Trainable: {param.requires_grad}")

        print("\n********** Text Decoder Trainable Status **********")
        for name, param in self.text_decoder.named_parameters():
            print(f"{name:60} Trainable: {param.requires_grad}")

        # # GRU layer
        # self.gru = nn.GRU(bert_config.hidden_size, bert_config.hidden_size, num_layers=num_layers, batch_first=True, dropout=(0 if gru_num_layers == 1 else dropout_rate))
        # self.dropout1 = nn.Dropout(dropout_rate)
        # self.classifier = nn.Linear(bert_config.hidden_size, 1)

        # BiLSTM layer
        # 1. 2. 3.
        # self.bilstm = nn.LSTM(bert_config.hidden_size, bert_config.hidden_size, num_layers=num_layers, batch_first=True, dropout=(0 if num_layers == 1 else dropout_rate), bidirectional=True)
        self.bilstm = nn.LSTM(768, 64, num_layers=num_layers, batch_first=True, dropout=(0 if num_layers == 1 else dropout_rate), bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        # # 1.
        # self.classifier = nn.Linear(bert_config.hidden_size * 2, 1)

        # 2.
        self.attention_layer = nn.Linear(64*2, 1)
        # self.classifier = nn.Linear(64*2, 1)

        self.connect_layer = nn.Linear(64*2, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier_lonely = nn.Linear(32, 1)
        self.classifier_sentiment = nn.Linear(32, 1)

        # # 3.
        # self.classifier = nn.Linear(bert_config.hidden_size * 2 + 1, 1)

        # # LSTM layer
        # self.lstm = nn.LSTM(bert_config.hidden_size, bert_config.hidden_size, num_layers=num_layers, batch_first=True, dropout=(0 if num_layers == 1 else dropout_rate), bidirectional=False)
        # self.dropout1 = nn.Dropout(dropout_rate)
        # self.classifier = nn.Linear(bert_config.hidden_size, 1)

        # # Attention mechanism
        # self.attention_weights = nn.Parameter(torch.randn(bert_config.hidden_size, 1))
        # self.attention_softmax = nn.Softmax(dim=1)
        #
        # # Feed-forward network
        # self.fc1 = nn.Linear(bert_config.hidden_size, 32)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.dropout2 = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(32, 16)
        # self.bn2 = nn.BatchNorm1d(16)
        # self.dropout3 = nn.Dropout(dropout_rate)
        # self.fc3 = nn.Linear(16, 8)
        # self.bn3 = nn.BatchNorm1d(8)
        # self.dropout4 = nn.Dropout(dropout_rate)
        # self.classifier = nn.Linear(8, 1)
        # self.relu = nn.ReLU()

    # Get sentiment (via VADER)
    def get_sentiment(self, texts):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = []
        for text in texts:
            vs = analyzer.polarity_scores(text)
            label = 1 if vs['compound'] > 0 else 0
            sentiment_scores.append(label)
        return sentiment_scores

    # Loss function
    def forward(self, index, prompt, label, reason, sentiment, device):
        # Get text embedding (analogous to prompt embedding)
        text = self.tokenizer(prompt, padding='max_length', truncation=True, return_tensors="pt").to(device)

        if self.configs['type'] == 'bert':
            text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        elif self.configs['type'] == 'roberta':
            text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)

        # ===============================================Binary Classification Loss (v ia entire BiLSTM + Attention sequence)===============================================
        text_embed = text_feat.last_hidden_state
        output, _ = self.bilstm(text_embed)
        output = self.dropout1(output)

        # # 1.
        # logits = self.classifier(output)
        # logits = logits.squeeze(-1)
        # logits_pooled = torch.mean(logits, dim=1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits_pooled, label)

        # 2.
        # ******************Binary Loss***********************
        attention_weights = torch.softmax(self.attention_layer(output), dim=1)
        attended_output = torch.sum(attention_weights * output, dim=1)
        # logits = self.classifier(attended_output)
        # logits = logits.squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)
        # loss_sentiment = torch.zeros(1)
        connected_output = self.connect_layer(attended_output)
        logits_lonely = self.classifier_lonely(connected_output).squeeze(-1)
        logits_sentiment = self.classifier_sentiment(connected_output).squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits_lonely, label)
        # loss_sentiment = F.binary_cross_entropy_with_logits(logits_sentiment, sentiment)
        loss_binary = focal_loss(logits_lonely, label, alpha=self.configs['alpha'], gamma=self.configs['gamma'])
        loss_sentiment = focal_loss(logits_sentiment, sentiment, alpha=1-self.configs['alpha'], gamma=self.configs['gamma'])

        # # Compute pairwise distances
        # dist_matrix = torch.cdist(attended_output, attended_output, p=2)
        # # Create a mask to ignore the diagonal (self-comparisons)
        # eye = torch.eye(dist_matrix.size(0)).bool().to(device)
        # dist_matrix = dist_matrix.masked_fill(eye, 0)
        # # Similarity labels matrix
        # label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
        # label_matrix = label_matrix.float().to(device)
        # # Define margin
        # margin = 1.0
        # # Calculate contrastive loss using the formula
        # positive_loss = (label_matrix * 0.5 * torch.pow(torch.clamp(margin - dist_matrix, min=0.0), 2)).mean()
        # negative_loss = ((1 - label_matrix) * 0.5 * torch.pow(dist_matrix, 2)).mean()
        # # Combine the losses
        # loss_constrast = positive_loss + negative_loss

        # ******************Constrastive Loss***********************
        embeddings_norm = F.normalize(attended_output, p=2, dim=1)
        cos_sim = torch.mm(embeddings_norm, embeddings_norm.t())
        match_loss = 0.5 * label * (1 - cos_sim) ** 2
        non_match_loss = 0.5 * (1 - label) * F.relu(self.configs['margin'] - (1 - cos_sim)) ** 2
        loss_constrast = match_loss + non_match_loss
        loss_constrast = loss_constrast.mean()

        # ******************Reason Loss***********************
        # # Add prompt to reason
        # reason = [self.configs['prompt'] + item for item in reason]

        # Tokenize and encode the reason
        if self.configs['type'] == 'bert':
            text_reason = self.tokenizer(reason, padding='max_length', truncation=True, return_tensors="pt", mode='text').to(device)
        elif self.configs['type'] == 'roberta':
            text_reason = self.tokenizer(reason, padding='max_length', truncation=True, return_tensors="pt").to(device)

        # Prepare decoder input using prompt's last hidden state
        decoder_input_ids = text_reason.input_ids.clone()
        bos_tokens = torch.full((decoder_input_ids.size(0), 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([bos_tokens, decoder_input_ids[:, :-1]], dim=1)
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        # Decode using reason
        outputs = self.text_decoder(input_ids=decoder_input_ids,
                                    attention_mask=text_reason.attention_mask,
                                    encoder_hidden_states=text_embed,
                                    encoder_attention_mask=text.attention_mask,
                                    labels=decoder_targets,
                                    return_dict=True)
        loss_reason = outputs.loss

        # # 3.
        # output_pooled = torch.mean(output, dim=1)
        # sentiment_features = sentiment.unsqueeze(-1)
        # combined_features = torch.cat((output_pooled, sentiment_features), dim=1)
        # logits = self.classifier(combined_features).squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)

        # # ===============================================Binary Classification Loss (via entire GRU sequence + Attention + FNN)===============================================
        # # Process sequence with GRU
        # text_embed = text_feat.last_hidden_state
        # output, _ = self.gru(text_embed)
        # output = self.dropout1(output)
        #
        # # Attention weights
        # attention_scores = torch.matmul(output, self.attention_weights)
        # attention_probs = self.attention_softmax(attention_scores)
        # weighted_output = output * attention_probs
        #
        # # Aggregate across the sequence using sum (as a form of attention-pooling)
        # aggregated_output = torch.sum(weighted_output, dim=1)
        #
        # # Process the output via feed-forward network
        # x = self.relu(self.bn1(self.fc1(aggregated_output)))
        # x = self.dropout2(x)
        # x = self.relu(self.bn2(self.fc2(x)))
        # x = self.dropout3(x)
        # x = self.relu(self.bn3(self.fc3(x)))
        # x = self.dropout4(x)
        # logits = self.classifier(x)
        #
        # # Compute binary cross-entropy loss
        # logits = logits.squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)

        # # ===============================================Binary Classification Loss (via entire LSTM sequence)===============================================
        # text_embed = text_feat.last_hidden_state
        # output, _ = self.lstm(text_embed)
        # output = self.dropout1(output)
        # logits = self.classifier(output)
        # logits = logits.squeeze(-1)
        # logits_pooled = torch.mean(logits, dim=1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits_pooled, label)

        # # ===============================================Binary Classification Loss (via entire GRU sequence)===============================================
        # text_embed = text_feat.last_hidden_state
        # output, _ = self.gru(text_embed)
        # output = self.dropout1(output)
        # logits = self.classifier(output)
        # logits = logits.squeeze(-1)
        # logits_avg = torch.mean(logits, dim=1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits_avg, label)

        # # ===============================================Binary Classification Loss (via hidden GRU sequence)===============================================
        # text_embed = text_feat.last_hidden_state
        # _, hidden = self.gru(text_embed)
        # hidden = self.dropout1(hidden)
        # logits = self.classifier(hidden[-1]).squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)

        # # ===============================================Binary Classification Loss (via linear layer)===============================================
        # text_embed = text_feat.last_hidden_state[:, 0, :]
        # logits = self.classifier(text_embed).squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)

        # # ===============================================Binary Classification Loss (via FNN)===============================================
        # text_embed = text_feat.last_hidden_state[:, 0, :]
        # logits = self.relu(self.bn1(self.fc1(text_embed)))
        # logits = self.dropout1(logits)
        # logits = self.relu(self.bn2(self.fc2(logits)))
        # logits = self.dropout2(logits)
        # logits = self.relu(self.bn3(self.fc3(logits)))
        # logits = self.dropout3(logits)
        # logits = self.fc4(logits)
        # loss_binary = F.binary_cross_entropy_with_logits(logits.squeeze(-1), label)
        return loss_binary, loss_sentiment, loss_constrast, loss_reason

    # Classify
    def classify(self, prompt, label, reason, sentiment, device):
        # Get text embedding (analogous to prompt embedding)
        text = self.tokenizer(prompt, padding='max_length', truncation=True, return_tensors="pt").to(device)

        if self.configs['type'] == 'bert':
            text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        elif self.configs['type'] == 'roberta':
            text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)

        # # ===============================================Binary Classification Loss (via entire GRU sequence + Attention + FNN)===============================================
        # # Process sequence with GRU
        # text_embed = text_feat.last_hidden_state
        # output, _ = self.gru(text_embed)
        # output = self.dropout1(output)
        #
        # # Attention weights
        # attention_scores = torch.matmul(output, self.attention_weights)
        # attention_probs = self.attention_softmax(attention_scores)
        # weighted_output = output * attention_probs
        #
        # # Aggregate across the sequence using sum (as a form of attention-pooling)
        # aggregated_output = torch.sum(weighted_output, dim=1)
        #
        # # Process the output via feed-forward network
        # x = self.relu(self.bn1(self.fc1(aggregated_output)))
        # x = self.dropout2(x)
        # x = self.relu(self.bn2(self.fc2(x)))
        # x = self.dropout3(x)
        # x = self.relu(self.bn3(self.fc3(x)))
        # x = self.dropout4(x)
        # logits = self.classifier(x)
        #
        # # Compute binary cross-entropy loss
        # logits = logits.squeeze(-1)
        # prob = torch.sigmoid(logits)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)

        # # ===============================================Binary Classification Loss (via entire LSTM sequence)===============================================
        # text_embed = text_feat.last_hidden_state
        # output, _ = self.lstm(text_embed)
        # output = self.dropout1(output)
        # logits = self.classifier(output)
        # logits = logits.squeeze(-1)
        # logits_pooled = torch.mean(logits, dim=1)
        # prob = torch.sigmoid(logits_pooled)
        # loss_binary = F.binary_cross_entropy_with_logits(logits_pooled, label)

        # ===============================================Binary Classification Loss (via entire BiLSTM sequence)===============================================
        text_embed = text_feat.last_hidden_state
        output, _ = self.bilstm(text_embed)
        output = self.dropout1(output)

        # # 1.
        # logits = self.classifier(output)
        # logits = logits.squeeze(-1)
        # logits_pooled = torch.mean(logits, dim=1)
        # prob = torch.sigmoid(logits_pooled)
        # loss_binary = F.binary_cross_entropy_with_logits(logits_pooled, label)

        # 2.
        # ******************Binary Loss***********************
        attention_weights = torch.softmax(self.attention_layer(output), dim=1)
        attended_output = torch.sum(attention_weights * output, dim=1)
        # logits = self.classifier(attended_output)
        # logits = logits.squeeze(-1)
        # prob = torch.sigmoid(logits)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)
        # loss_sentiment = torch.zeros(1)
        connected_output = self.connect_layer(attended_output)
        logits_lonely = self.classifier_lonely(connected_output).squeeze(-1)
        logits_sentiment = self.classifier_sentiment(connected_output).squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits_lonely, label)
        # loss_sentiment = F.binary_cross_entropy_with_logits(logits_sentiment, sentiment)
        loss_binary = focal_loss(logits_lonely, label, alpha=self.configs['alpha'], gamma=self.configs['gamma'])
        loss_sentiment = focal_loss(logits_sentiment, sentiment, alpha=1-self.configs['alpha'], gamma=self.configs['gamma'])
        prob = torch.sigmoid(logits_lonely)

        # ******************Constrastive Loss***********************
        embeddings_norm = F.normalize(attended_output, p=2, dim=1)
        cos_sim = torch.mm(embeddings_norm, embeddings_norm.t())
        match_loss = 0.5 * label * (1 - cos_sim) ** 2
        non_match_loss = 0.5 * (1 - label) * F.relu(self.configs['margin'] - (1 - cos_sim)) ** 2
        loss_constrast = match_loss + non_match_loss
        loss_constrast = loss_constrast.mean()

        # ******************Reason Loss***********************
        # # Add prompt
        # reason = [self.configs['prompt'] + item for item in reason]

        # Tokenize and encode the reason
        if self.configs['type'] == 'bert':
            text_reason = self.tokenizer(reason, padding='max_length', truncation=True, return_tensors="pt", mode='text').to(device)
        elif self.configs['type'] == 'roberta':
            text_reason = self.tokenizer(reason, padding='max_length', truncation=True, return_tensors="pt").to(device)

        # Prepare decoder input using prompt's last hidden state
        decoder_input_ids = text_reason.input_ids.clone()
        bos_tokens = torch.full((decoder_input_ids.size(0), 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([bos_tokens, decoder_input_ids[:, :-1]], dim=1)
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        # Decode using reason
        outputs = self.text_decoder(input_ids=decoder_input_ids,
                                    attention_mask=text_reason.attention_mask,
                                    encoder_hidden_states=text_embed,
                                    encoder_attention_mask=text.attention_mask,
                                    labels=decoder_targets,
                                    return_dict=True)
        loss_reason = outputs.loss

        # # 3.
        # output_pooled = torch.mean(output, dim=1)
        # sentiment_features = sentiment.unsqueeze(-1)
        # combined_features = torch.cat((output_pooled, sentiment_features), dim=1)
        # logits = self.classifier(combined_features).squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)
        # prob = torch.sigmoid(logits)

        # # ===============================================Binary Classification Loss (via entire GRU sequence)===============================================
        # text_embed = text_feat.last_hidden_state
        # output, _ = self.gru(text_embed)
        # output = self.dropout1(output)
        # logits = self.classifier(output)
        # logits = logits.squeeze(-1)
        # logits_avg = torch.mean(logits, dim=1)
        # prob = torch.sigmoid(logits_avg)
        # loss_binary = F.binary_cross_entropy_with_logits(logits_avg, label)

        # # ===============================================Binary Classification (via hidden GRU sequence)===============================================
        # text_embed = text_feat.last_hidden_state
        # _, hidden = self.gru(text_embed)
        # logits = self.classifier(hidden[-1]).squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)
        # prob = torch.sigmoid(logits)

        # # ===============================================Binary Classification Loss (via linear layer)===============================================
        # text_embed = text_feat.last_hidden_state[:, 0, :]
        # logits = self.classifier(text_embed).squeeze(-1)
        # loss_binary = F.binary_cross_entropy_with_logits(logits, label)
        # prob = torch.sigmoid(logits)

        # # ===============================================Binary Classification Loss (via FNN)===============================================
        # text_embed = text_feat.last_hidden_state[:, 0, :]
        # logits = self.relu(self.bn1(self.fc1(text_embed)))
        # logits = self.relu(self.bn2(self.fc2(logits)))
        # logits = self.relu(self.bn3(self.fc3(logits)))
        # logits = self.fc4(logits)
        # loss_binary = F.binary_cross_entropy_with_logits(logits.squeeze(-1), label)
        # prob = torch.sigmoid(logits)
        return loss_binary, loss_sentiment, loss_constrast, loss_reason, prob

    def generate(self, prompt, device, max_length=128, min_length=10, top_p=0.9):
        # Tokenize the input prompt
        text = self.tokenizer(prompt, max_length=max_length, truncation=True, padding='max_length', return_tensors="pt").to(device)

        # Get embeddings from text encoder
        if self.configs['type'] == 'bert':
            text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        elif self.configs['type'] == 'roberta':
            text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)

        encoder_hidden_states = text_feat.last_hidden_state

        # Attention masks for the inputs
        # encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long).to(self.configs['eval_device'])
        encoder_attention_mask = text.attention_mask

        # Prompt
        prompt = [self.configs['prompt']] * len(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Set up bos_id
        bos_tokens = torch.full((input_ids.shape[0], 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([bos_tokens, input_ids[:, :-1]], dim=1)

        # nucleus sampling
        outputs = self.text_decoder.generate(input_ids=decoder_input_ids,
                                             max_length=max_length,
                                             min_length=min_length,
                                             do_sample=True,
                                             top_p=top_p,
                                             num_return_sequences=1,
                                             eos_token_id=self.tokenizer.sep_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             repetition_penalty=1.1,
                                             encoder_hidden_states=encoder_hidden_states,
                                             encoder_attention_mask=encoder_attention_mask)

        # Decode generated output tokens to strings
        reasons = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return reasons

# Initialize BertLonely model
def init_bert_lonely(pretrained, **kwargs):
    model = BertLonely(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model
