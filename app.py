import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
import sklearn
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import TFDistilBertModel, AdamWeightDecay
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import tensorflow_addons as tfa

from flask import Flask, request, jsonify, render_template
application = Flask(__name__)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

with custom_object_scope({'TFDistilBertModel': TFDistilBertModel, 'AdamWeightDecay': AdamWeightDecay}):
    model = load_model('saved_model/best_model.h5')

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def remove_stopwords_and_punctuations(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    words_without_punctuations = [''.join(c for c in word if c not in string.punctuation) for word in words]
    words_preprocessed = [word.replace("‘", "").replace("’", "").replace("“"," ").replace("”"," ") for word in words_without_punctuations if len(word)>2]
    return ' '.join(words_preprocessed)

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids'])


def predict_category( headline, short_description ):

    # Combine the headline and short_description
    text = headline + ' ' + short_description

    # Preprocess the text
    text = remove_stopwords_and_punctuations(text)

    # Tokenize and encode the text
    encoded_text = regular_encode([text], tokenizer, maxlen=512)

    # Convert to TensorFlow dataset
    dataset = (
        tf.data.Dataset
        .from_tensor_slices(encoded_text)
        .batch(1)
    )

    # Make prediction
    pred = model.predict(dataset, verbose=1)
    pred_class = np.argmax(pred, axis=1)

    # Decode the predicted class
    # category = encoder.inverse_transform(pred_class)
    pred_number=pred_class[0]
    
    category_dict={'ARTS & CULTURE': 0, 'BUSINESS & FINANCES': 1, 'COMEDY': 2, 'CRIME': 3, 'DIVORCE': 4, 'EDUCATION': 5, 'ENTERTAINMENT': 6, 'ENVIRONMENT': 7, 'FOOD & DRINK': 8, 'GROUPS VOICES': 9, 'HOME & LIVING': 10, 'IMPACT': 11, 'MEDIA': 12, 'MISCELLANEOUS': 13, 'PARENTING': 14, 'POLITICS': 15, 'RELIGION': 16, 'SCIENCE & TECH': 17, 'SPORTS': 18, 'STYLE & BEAUTY': 19, 'TRAVEL': 20, 'U.S. NEWS': 21, 'WEDDINGS': 22, 'WEIRD NEWS': 23, 'WELLNESS': 24, 'WOMEN': 25, 'WORLD NEWS': 26}
    category = None
    category=[key for key,value in category_dict.items() if value==pred_number]

    if category is not None:
        return category[0]
    else:
        # Handle the case when category_list is None
        # For example, raise an exception or return a default value
        raise ValueError(f"No matching value found for category")



# A dummy prediction here to "warm up" the model
dummy_text = "This is a dummy text for model warm up"
dummy_encoded_text = regular_encode([dummy_text], tokenizer, maxlen=512)
dummy_dataset = tf.data.Dataset.from_tensor_slices(dummy_encoded_text).batch(1)
_ = model.predict(dummy_dataset, verbose=0)
print("Dummy code executed")



#This function returns the home page of the web application
@application.route('/')
def home():
    return render_template('index.html')

#This function takes the user’s input from a form and predicts the news category
@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    title= features[0]
    paragraph=features[1]
    prediction_value = predict_category(title,paragraph)
    return render_template('index.html', prediction_text='News Category Should Be {}'.format(prediction_value))

#This function handles API requests and returns a JSON response with the prediction value
@application.route("/api", methods=["GET", "POST"])
def api_predict():
    '''
    For handling API calls
    '''
    data = request.get_json(force=True)
    title = data['title']
    paragraph = data['description']
    prediction_value = predict_category(title, paragraph)
    return jsonify({'prediction': prediction_value})

# This line checks if this script is being run directly or being imported
# If the script is being run directly, then __name__ will be "__main__"
if __name__ == "__main__":
    # This line starts the Flask application
    # host='0.0.0.0' makes the server publicly available across the network
    # port=8000 is the port number where the server will be listening
    application.run(host='0.0.0.0', port=8000)
