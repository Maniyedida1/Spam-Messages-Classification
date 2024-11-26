# Required Libraries
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Create the Streamlit app
st.title('Predicting if a Message is Spam or Not')

# Sample Data to Train a Simple Model
data = pd.DataFrame({
    'message': ['free money now', 'call me later', 'win a lottery', 'how are you', 'claim your prize'],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam']
})

# Preparing the Data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train the Model
model = MultinomialNB()
model.fit(X, y)

# Input from User
message = st.text_input('Enter a message')

submit = st.button('Predict')

if submit:
    # Transform the input message
    message_transformed = vectorizer.transform([message])
    
    # Make prediction
    prediction = model.predict(message_transformed)
    
    # Display the result
    if prediction[0] == 'spam':
        st.warning('This message is spam')
    else:
        st.success('This message is Legit (HAM)')
