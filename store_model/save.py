import torch

from transformers import AutoModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM

from utils.system import get_store_model

if __name__ == "__main__":
    # Save bert base uncased pth file
    print("Saving bert base uncased pth file")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    torch.save(model.state_dict(), get_store_model() / 'bert_base_uncased.pth')

    # Save bert reddit finetune
    print("Saving mental bert finetune pth file")
    model = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
    torch.save(model.state_dict(), get_store_model() / 'bert_mental_base_uncased.pth')
