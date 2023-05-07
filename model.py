import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import re

model_bert = SentenceTransformer('all-MiniLM-L6-v2')

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)


with open("multilabel_binarizer.pkl", "rb") as f:
    classes = pickle.load(f)

def predict_pipeline(text):
    my_punctuation = '!"$%&\()*,-./:;<=>?@[\\]^_`{|}~'
    #text lowercase
    text = text.lower()
    #remove punctiation in except + and # for C++ and C#
    text = text.translate(str.maketrans("", "", my_punctuation))
    #remove numbers
    text = re.sub('\d+', '', text)
    # Remove unicode characters
    text = text.encode("ascii", "ignore").decode()
    # Remove extra spaces
    text = re.sub('\s+', ' ', text)
    #tokenization
    #text = text.split()
    #remove_stop_words
    #text = [word for word in text.split() if not word in stop_words]
    sent = []
    sent.append(text)
    sentence_embeddings = model_bert.encode(sent)
    y_pred = model.predict(sentence_embeddings)
    tags = classes.inverse_transform(y_pred)
    tag = [item for t in tags for item in t]
    return tag
