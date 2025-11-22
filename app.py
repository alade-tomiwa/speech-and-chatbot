import streamlit as st
import nltk
import numpy as np
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import string

# --- CONFIGURATION ---
st.set_page_config(page_title="Arsenal Bot", page_icon="⚽")
st.title("🔴⚪ Smart Arsenal Chatbot")

# --- 1. SETUP & DOWNLOADS ---
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    return nltk.stem.WordNetLemmatizer()

lemmer = setup_nltk()

# --- 2. SMART DATA LOADING & CLEANING ---
def clean_text_data(raw_text):
    """
    This function cleans the dirty text file.
    It removes headlines, short fragments, and junk data.
    """
    cleaned_sentences = []
    # Split by lines first to remove "menu items" or "titles"
    lines = raw_text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Filter: Must be at least 5 words and end with punctuation
        if len(line.split()) > 4 and line[-1] in ['.', '!', '?']:
            cleaned_sentences.append(line)
            
    return " ".join(cleaned_sentences)

# Load and Process Data
if os.path.exists("arsenal.txt"):
    with open("arsenal.txt", "r", errors='ignore') as f:
        raw_data_dirty = f.read()
    
    # Apply the cleaner
    raw_data = clean_text_data(raw_data_dirty)
    raw_data = raw_data.lower()
    
    # Create tokens
    sent_tokens = nltk.sent_tokenize(raw_data)
    word_tokens = nltk.word_tokenize(raw_data)
else:
    st.error("CRITICAL ERROR: 'arsenal.txt' file not found.")
    st.stop()

# --- 3. PRE-PROCESSING FUNCTIONS ---
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# --- 4. RESPONSE ENGINE (TF-IDF) ---
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    
    # Calculate similarity
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    
    # Find best match
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    # Logic: If similarity is 0, I don't know.
    if(req_tfidf == 0):
        robo_response = "I am sorry, I don't have information about that in my text file."
    else:
        robo_response = sent_tokens[idx]
        
    sent_tokens.remove(user_response)
    return robo_response
# --- SAFE SPEECH FUNCTION ---
def transcribe_speech():
    r = sr.Recognizer()
    try:
        # We try to access the microphone
        with sr.Microphone() as source:
            st.info("Listening... Speak now!")
            r.adjust_for_ambient_noise(source)
            try:
                audio_text = r.listen(source, timeout=5)
                st.info("Transcribing...")
                text = r.recognize_google(audio_text)
                return text
            except sr.UnknownValueError:
                return "Sorry, I did not understand that."
            except sr.RequestError:
                return "Sorry, speech service is down."
            except sr.WaitTimeoutError:
                return "No speech detected."
    except OSError:
        # This error happens on Streamlit Cloud because there is no mic
        return "ERROR: No Microphone detected. (This feature only works on Localhost, not Cloud)."
    except Exception as e:
        return f"An error occurred: {e}"

# --- 7. MAIN INTERFACE ---
st.markdown("#### Ask me about the Stadium, Manager, or Players.")

tab1, tab2 = st.tabs(["⌨️ Type", "🎤 Speak"])

user_input = ""

with tab1:
    text_in = st.text_input("Type your question here:")
    if st.button("Send"):
        user_input = text_in

with tab2:
    if st.button("Start Recording"):
        speech_out = transcribe_speech()
        if "Sorry" not in speech_out and "No speech" not in speech_out:
            st.success(f"You said: '{speech_out}'")
            user_input = speech_out
        else:
            st.error(speech_out)

# --- 8. GENERATE ANSWER ---
if user_input:
    user_input = user_input.lower()
    
    # Hardcoded greetings for better UX
    if user_input in ['hi', 'hello', 'hey']:
        st.write("🤖 **Bot:** Hello! Ask me about Arsenal.")
    elif user_input in ['thanks', 'thank you']:
        st.write("🤖 **Bot:** You are welcome!")
    else:
        # AI Generation
        with st.spinner("Thinking..."):
            result = response(user_input)

            st.markdown(f"### 🤖 Bot says:\n> {result.capitalize()}")
