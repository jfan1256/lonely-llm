import re
import unicodedata

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text).strip()
    contractions = {
        "can't": "cannot", "didn't": "did not", "don't": "do not",
        "it's": "it is", "i'm": "i am", "you're": "you are",
        "he's": "he is", "she's": "she is", "that's": "that is",
        "there's": "there is", "what's": "what is", "who's": "who is"
    }
    words = text.split()
    reformed = [contractions[word] if word in contractions else word for word in words]
    text = " ".join(reformed)
    return text