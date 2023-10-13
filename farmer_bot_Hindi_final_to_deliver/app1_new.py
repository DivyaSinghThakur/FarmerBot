import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
nltk.download("indian")
from keras.models import load_model
model = load_model('D:\\farmer_bot_Hindi_final_to_deliver\\farmer_bot_Hindi_final_to_deliver\\model_exp.h5')
import json
import random
intents = json.loads(open('D:\\farmer_bot_Hindi_final_to_deliver\\farmer_bot_Hindi_final_to_deliver\\hindi_responce.json', encoding='utf-8').read())
words = pickle.load(open('D:\\farmer_bot_Hindi_final_to_deliver\\farmer_bot_Hindi_final_to_deliver\\texts.pkl','rb'))
classes = pickle.load(open('D:\\farmer_bot_Hindi_final_to_deliver\\farmer_bot_Hindi_final_to_deliver\\labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    # print(ints)
    list_of_intents = intents_json['intents']
    # print(list_of_intents)
    tag_match=False
    for i in list_of_intents:
        # print(tag)
        # print(i["tag"])
        if(i['tag']== tag and float(ints[0]['probability'])>.9):
            print(tag)
            result = random.choice(i['responses'])
            tag_match=True
            break

    if tag_match==True:
        return result
            
    else:
        result="मुझे खेद है, लेकिन मुझे आपकी क्वेरी के लिए उपयुक्त उत्तर नहीं मिला। मैंने इसे आगे की सहायता के लिए नॉलेज क्यूरेशन सेंटर (केसीसी) में हमारी टीम को भेज दिया है। "
        # print(result)
        return result 

def chatbot_response(msg):
    file_path="hindi_msg.txt"
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write("\n")
        file.write(msg)
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()