import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from class_models.loss import focal_loss, tversky_loss, dice_loss
from class_models.bert import BertConfig, BertModel, BertLMHeadModel
from class_models.utils import load_checkpoint, tie_encoder_decoder_weights, set_trainable

class BertLonely(nn.Module):
    def __init__(self,
                 configs,
                 ):
        super().__init__()

        # Configs
        self.configs = configs

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]

        # Text Encoder
        bert_config = BertConfig.from_json_file(self.configs['bert_config'])
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        total = sum([param.nelement() for param in self.text_encoder.parameters()])
        print('Text Encoder Number of Params: %.2fM' % (total / 1e6))

        # Text Decoder
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=bert_config)
        total = sum([param.nelement() for param in self.text_decoder.parameters()])
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        print('Text Decoder Number of Params: %.2fM' % (total / 1e6))

        # Tie Encoder and Decoder
        print("\n********** Tie Encoder and Decoder **********")
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')

        # Freeze parameters of the text_encoder and text_decoder to not train them
        encoder_layers_to_train = ['encoder.layer.11.']
        decoder_layers_to_train = ['bert.encoder.layer.11.']
        set_trainable(model_component=self.text_encoder, layer_names=encoder_layers_to_train, type='encoder', include_predictions=False, include_embeddings=False)
        set_trainable(model_component=self.text_decoder, layer_names=decoder_layers_to_train, type='decoder', include_predictions=True, include_embeddings=False)

        print("\n********** Text Encoder Trainable Status **********")
        for name, param in self.text_encoder.named_parameters():
            print(f"{name:60} Trainable: {param.requires_grad}")

        print("\n********** Text Decoder Trainable Status **********")
        for name, param in self.text_decoder.named_parameters():
            print(f"{name:60} Trainable: {param.requires_grad}")

        # BiLSTM layer
        self.bilstm = nn.LSTM(768, 64, num_layers=1, batch_first=True, bidirectional=True)

        # Multi-head Attention Layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=64*2, num_heads=self.configs['num_head_classifier'], batch_first=True)
        self.attention_layer = nn.Linear(64*2, 1)

        # Dropout
        self.dropout1 = nn.Dropout(configs['dropout_rate'])
        self.dropout2 = nn.Dropout(configs['dropout_rate'])

        # Classifier
        self.connect_layer = nn.Linear(64*2, 32)
        self.classifier_lonely = nn.Linear(32, 1)
        self.classifier_sentiment = nn.Linear(32, 1)

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
        text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')

        # ===============================================Binary Classification Loss (via entire BiLSTM + Attention sequence)===============================================
        # Feed in BiLSTM
        text_embed = text_feat.last_hidden_state
        output, _ = self.bilstm(text_embed)
        output = self.dropout1(output)

        # Apply Multi-head Attention
        output, _ = self.multihead_attention(output, output, output)
        output = self.dropout2(output)
        attention_weights = torch.softmax(self.attention_layer(output), dim=1)
        attended_output = torch.sum(attention_weights * output, dim=1)

        # Connection layer
        connected_output = self.connect_layer(attended_output)

        # Get lonely and sentiment logits
        logits_lonely = self.classifier_lonely(connected_output).squeeze(-1)
        logits_sentiment = self.classifier_sentiment(connected_output).squeeze(-1)
        prob = torch.sigmoid(logits_lonely)

        # ******************Focal Loss***********************
        loss_lonely = focal_loss(logits_lonely, label, alpha=self.configs['alpha_focal'], gamma=self.configs['gamma_focal'])
        loss_sentiment = focal_loss(logits_sentiment, sentiment, alpha=1-self.configs['alpha_focal'], gamma=self.configs['gamma_focal'])

        # ******************Focal Loss***********************
        loss_dice = dice_loss(logits_lonely, label)

        # ******************Tversky Loss***********************
        loss_tversky = tversky_loss(logits_lonely, label, alpha=self.configs['alpha_tverksy'], beta=self.configs['beta_tversky'])

        # ******************Constrastive Loss***********************
        embeddings_norm = F.normalize(attended_output, p=2, dim=1)
        cos_sim = torch.mm(embeddings_norm, embeddings_norm.t())
        match_loss = 0.5 * label * (1 - cos_sim) ** 2
        non_match_loss = 0.5 * (1 - label) * F.relu(self.configs['margin'] - (1 - cos_sim)) ** 2
        loss_constrast = torch.mean(match_loss + non_match_loss)

        # ******************Reason Loss***********************
        # Add prompt to reason
        reason = [self.configs['prompt'] + item for item in reason]

        # Tokenize and encode the reason
        text_reason = self.tokenizer(reason, padding='max_length', truncation=True, return_tensors="pt", mode='text').to(device)

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

        loss_lonely = torch.zeros(1, device=device)
        loss_sentiment = torch.zeros(1, device=device)
        return loss_lonely, loss_sentiment, loss_dice, loss_tversky, loss_constrast, loss_reason, prob

    # Classify
    def classify(self, prompt, device):
        # Get text embedding (analogous to prompt embedding)
        text = self.tokenizer(prompt, padding='max_length', truncation=True, return_tensors="pt").to(device)
        text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_embed = text_feat.last_hidden_state
        output, _ = self.bilstm(text_embed)
        output = self.dropout1(output)
        attention_weights = torch.softmax(self.attention_layer(output), dim=1)
        attended_output = torch.sum(attention_weights * output, dim=1)
        connected_output = self.connect_layer(attended_output)
        logits_lonely = self.classifier_lonely(connected_output).squeeze(-1)
        prob = torch.sigmoid(logits_lonely)
        return prob

    def generate(self, prompt, device, max_length=128, min_length=10, top_p=0.9):
        # Tokenize the input prompt
        text = self.tokenizer(prompt, max_length=max_length, truncation=True, padding='max_length', return_tensors="pt").to(device)
        text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')

        # Hidden States
        encoder_hidden_states = text_feat.last_hidden_state

        # Attention masks
        encoder_attention_mask = text.attention_mask

        # Prompt
        prompt = [self.configs['prompt']] * len(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Set up bos_id
        bos_tokens = torch.full((input_ids.shape[0], 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([bos_tokens, input_ids[:, :-1]], dim=1)

        # Nucleus sampling
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