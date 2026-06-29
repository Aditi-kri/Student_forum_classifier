import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

def preprocess(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    words = [ps.stem(word)
             for word in words
             if word not in stop_words]

    return " ".join(words)