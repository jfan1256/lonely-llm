import torch

from torch import nn
from transformers import BertTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from class_models.bert import BertConfig, BertModel, BertLMHeadModel
from class_models.model_utils import load_checkpoint, tie_encoder_decoder_weights, set_trainable
from class_models.loss import focal_loss, tversky_loss, dice_loss, center_loss, contrast_loss_encoder, large_margin_cosine_loss, contrast_loss_decoder, embed_match_loss, perplex_loss

class BertLonely(nn.Module):
    def __init__(self,
                 configs,
                 ):
        super().__init__()

        # Configs
        self.configs = configs

        # Tokenizer
        if configs['bert_model'] == 'mental/mental-bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained(configs['bert_model'])
            self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
            self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
            self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]

        # Text Encoder
        bert_config = BertConfig.from_json_file(self.configs['bert_config'])
        self.text_encoder = BertModel.from_pretrained(configs['bert_model'], config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        total = sum([param.nelement() for param in self.text_encoder.parameters()])
        print('Text Encoder Number of Params: %.2fM' % (total / 1e6))

        # Text Decoder
        self.text_decoder = BertLMHeadModel.from_pretrained(configs['bert_model'], config=bert_config)
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

        # Multilayer Perceptron for loneliness
        self.mlp_lonely = nn.Sequential(
            nn.Linear(bert_config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Multilayer Perceptron for sentiment
        self.mlp_sentiment = nn.Sequential(
            nn.Linear(bert_config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    # Get sentiment (via VADER)
    def get_sentiment(self, texts):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = []
        for text in texts:
            vs = analyzer.polarity_scores(text)
            label = 1 if vs['compound'] > 0 else 0
            sentiment_scores.append(label)
        return sentiment_scores

    # Train
    def forward(self, index, narrative, label, reason, sentiment, device):
        # Get text embedding (analogous to narrative embedding)
        text = self.tokenizer(narrative, padding='max_length', truncation=True, return_tensors="pt").to(device)
        text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_embed = text_feat.last_hidden_state

        # Extract the [CLS] token's final layer features
        enc_cls_output = text_feat.last_hidden_state[:, 0]

        # Classify lonely
        logits_lonely = self.mlp_lonely(enc_cls_output).squeeze(-1)

        # Classify sentiment
        logits_sentiment = self.mlp_sentiment(enc_cls_output).squeeze(-1)

        # ===============================================Binary Classification Loss (via MLP)===============================================
        # ******************Focal Loss***********************
        if 'loss_focal' in self.configs['loss']:
            loss_focal_lonely = focal_loss(logits_lonely, label, alpha=self.configs['alpha_focal'], gamma=self.configs['gamma_focal'])
            loss_focal_sentiment = focal_loss(logits_sentiment, sentiment, alpha=1-self.configs['alpha_focal'], gamma=self.configs['gamma_focal'])
            loss_focal = self.configs['alpha'] * loss_focal_lonely + (1 - self.configs['alpha']) * loss_focal_sentiment
        else:
            loss_focal = torch.zeroes(1)

        # ******************Dice Loss***********************
        if 'loss_dice' in self.configs['loss']:
            loss_dice_lonely = dice_loss(logits_lonely, label)
            loss_dice_sentiment = dice_loss(logits_sentiment, sentiment)
            loss_dice = self.configs['alpha'] * loss_dice_lonely + (1 - self.configs['alpha']) * loss_dice_sentiment
        else:
            loss_dice = torch.zeroes(1)

        # ******************Tversky Loss***********************
        if 'loss_tversky' in self.configs['loss']:
            loss_tversky_lonely = tversky_loss(logits_lonely, label, alpha=self.configs['alpha_tverksy'], beta=self.configs['beta_tversky'])
            loss_tversky_sentiment = tversky_loss(logits_sentiment, sentiment, alpha=self.configs['alpha_tverksy'], beta=self.configs['beta_tversky'])
            loss_tversky = self.configs['alpha'] * loss_tversky_lonely + (1 - self.configs['alpha']) * loss_tversky_sentiment
        else:
            loss_dice = torch.zeroes(1)

        # ******************Center Loss***********************
        if 'loss_center' in self.configs['loss']:
            loss_center_lonely = center_loss(enc_cls_output, label, self.centers, alpha=self.configs['alpha_center'])
            loss_center_sentiment = center_loss(enc_cls_output, sentiment, self.centers, alpha=self.configs['alpha_center'])
            loss_center = self.configs['alpha'] * loss_center_lonely + (1 - self.configs['alpha']) * loss_center_sentiment
        else:
            loss_center = torch.zeros(1, device=device)

        # ******************Angular Loss***********************
        if 'loss_angular' in self.configs['loss']:
            loss_angular_lonely = large_margin_cosine_loss(enc_cls_output, label, margin=self.configs['margin_angular'])
            loss_angular_sentiment = large_margin_cosine_loss(enc_cls_output, sentiment, margin=self.configs['margin_angular'])
            loss_angular = self.configs['alpha'] * loss_angular_lonely + (1 - self.configs['alpha']) * loss_angular_sentiment
        else:
            loss_angular = torch.zeros(1, device=device)

        # ******************Reason Loss***********************
        # Add prompt to reason
        reason = [self.configs['prompt'] + item for item in reason]

        # Tokenize and encode the reason
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
                                    return_dict=True,
                                    output_attentions=True,
                                    output_hidden_states=True,
                                    )

        # Decoder CLS output
        dec_cls_output = outputs.hidden_states[-1][:, 0]

        # LM Loss
        if 'loss_reason' in self.configs['loss']:
            loss_reason = outputs.loss
        else:
            loss_reason =  torch.zeros(1, device=self.configs['train_device'])

        # ******************Constrastive Loss***********************
        if 'loss_contrast' in self.configs['loss']:
            loss_contrast_lonely = contrast_loss_encoder(enc_cls_output, label, margin=self.configs['margin_contrast'])
            loss_contrast_sentiment = contrast_loss_encoder(enc_cls_output, sentiment, margin=self.configs['margin_contrast'])
            loss_contrast_enc = self.configs['alpha'] * loss_contrast_lonely + (1 - self.configs['alpha']) * loss_contrast_sentiment
            loss_contrast = loss_contrast_enc
            # loss_contrast_dec = contrast_loss_decoder(enc_cls_output, dec_cls_output, label, margin=self.configs['margin_contrast'])
            # loss_contrast = loss_contrast_enc + loss_contrast_dec
        else:
            loss_contrast =  torch.zeros(1, device=self.configs['train_device'])

        # ******************Perplexity Loss***********************
        if 'loss_perplex' in self.configs['loss']:
            loss_perplex = perplex_loss(outputs.logits, decoder_targets)
        else:
            loss_perplex =  torch.zeros(1, device=self.configs['train_device'])

        # ******************Embedding Match Loss***********************
        if 'loss_embed_match' in self.configs['loss']:
            text_reason_feat = self.text_encoder(text_reason.input_ids, attention_mask=text_reason.attention_mask, return_dict=True, mode='text')
            reason_enc_cls_output = text_reason_feat.last_hidden_state[:, 0]
            loss_embed_match = embed_match_loss(dec_cls_output, reason_enc_cls_output)
        else:
            loss_embed_match =  torch.zeros(1, device=self.configs['train_device'])

        # Return
        return {
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_tversky': loss_tversky,
            'loss_center': loss_center,
            'loss_angular': loss_angular,
            'loss_contrast': loss_contrast,
            'loss_reason': loss_reason,
            'loss_perplex': loss_perplex,
            'loss_embed_match': loss_embed_match
        }

    # Classify
    def classify(self, narrative, device):
        text = self.tokenizer(narrative, padding='max_length', truncation=True, return_tensors="pt").to(device)
        text_feat = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        cls_output = text_feat.last_hidden_state[:, 0]
        logits_lonely = self.mlp_lonely(cls_output).squeeze(-1)
        prob = torch.sigmoid(logits_lonely)
        return prob

    # Generate
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