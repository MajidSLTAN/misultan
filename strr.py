import numpy as np
import speech_recognition as sr
import os
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.preprocessing import LabelEncoder
import cv2
from gtts import gTTS
import pygame
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from snowballstemmer import stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import dlib
from skimage import io
import streamlit as st
import time
import gdown

# تنزيل حزم NLTK المطلوبة
nltk.download('punkt')
nltk.download('stopwords')

# إعدادات معالجة اللغة العربية
ar_stemmer = stemmer("arabic")
arabic_stopwords = set(stopwords.words('arabic'))

# تحميل الملفات من Google Drive
def download_files_from_drive():
    if not os.path.exists('./model'):
        os.makedirs('./model')

    model_url = "https://drive.google.com/uc?id=FILE_ID_MODEL"
    vectorizer_url = "https://drive.google.com/uc?id=FILE_ID_VECTORIZER"
    label_encoder_url = "https://drive.google.com/uc?id=FILE_ID_LABEL_ENCODER"
    face_model_url = "https://drive.google.com/uc?id=FILE_ID_FACE_MODEL"
    shape_predictor_url = "https://drive.google.com/uc?id=FILE_ID_SHAPE_PREDICTOR"

    if not os.path.exists('./model/mbotGRU_model8.h5'):
        gdown.download(model_url, './model/mbotGRU_model8.h5', quiet=False)
    if not os.path.exists('tfidf_vectorizer8.pkl'):
        gdown.download(vectorizer_url, 'tfidf_vectorizer8.pkl', quiet=False)
    if not os.path.exists('label_encodergrum8.pkl'):
        gdown.download(label_encoder_url, 'label_encodergrum8.pkl', quiet=False)
    if not os.path.exists('./model/dlib_face_recognition_resnet_model_v1.dat'):
        gdown.download(face_model_url, './model/dlib_face_recognition_resnet_model_v1.dat', quiet=False)
    if not os.path.exists('./model/shape_predictor_68_face_landmarks.dat'):
        gdown.download(shape_predictor_url, './model/shape_predictor_68_face_landmarks.dat', quiet=False)

# تحميل النموذج والمتجهات
download_files_from_drive()
model = tf.keras.models.load_model('./model/mbotGRU_model8.h5')
with open('tfidf_vectorizer8.pkl', 'rb') as file:
    tfidf = pickle.load(file)
with open('label_encodergrum8.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# تحميل الردود المكتسبة والأشخاص المعروفين
learned_responses_file = 'learned_responses.json'
learned_responses = json.load(open(learned_responses_file, 'r', encoding='utf-8')) if os.path.exists(learned_responses_file) else {}
known_people_file = 'known_people.json'
known_people = {}
if os.path.exists(known_people_file):
    with open(known_people_file, 'r', encoding='utf-8') as file:
        known_people = json.load(file)

# تنظيف النص
def cleaning_text(text):
    text = re.sub(r'[\n\W]', ' ', text)
    text = ''.join(c for c in text if c not in string.punctuation)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\ة', 'ه', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in arabic_stopwords]
    return ' '.join(tokens)

# حفظ الردود المكتسبة
def save_learned_response(question, answer):
    learned_responses[question] = answer
    with open(learned_responses_file, 'w', encoding='utf-8') as file:
        json.dump(learned_responses, file, ensure_ascii=False, indent=4)

# حفظ الأشخاص المعروفين
def save_known_person(name, embedding):
    if name in known_people:
        known_people[name].append(embedding)
    else:
        known_people[name] = [embedding]
    with open(known_people_file, 'w', encoding='utf-8') as file:
        json.dump(known_people, file, ensure_ascii=False, indent=4)

# تحميل بيانات الأشخاص المعروفين
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('./model/dlib_face_recognition_resnet_model_v1.dat')
current_person = None

# استخراج وحفظ تضمين الوجه
def extract_and_save_face_embedding(name, face_img):
    dets = face_detector(face_img, 1)
    if len(dets) > 0:
        shape = shape_predictor(face_img, dets[0])
        embedding = face_rec_model.compute_face_descriptor(face_img, shape)
        embedding = np.array(embedding).tolist()
        save_known_person(name, embedding)
def extract_name(input_text):
    # إزالة العبارات الزائدة مثل "أنا اسمي"، "اسمي هو"، "أنا"
    patterns = [
        r'\b(?:انا|اسمي هو|اسمي|أناانا اسمي|انا اسمي|)\b',  # العبارات الشائعة للتعريف بالنفس
    ]
    for pattern in patterns:
        input_text = re.sub(pattern, '', input_text, flags=re.IGNORECASE).strip()
    return input_text

# التعرف على الشخص
previous_person = None
similarity_threshold = 0.9
success_count = 0
failure_count = 0

def recognize_person():
    global current_person, previous_person, similarity_threshold, success_count, failure_count
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = face_detector(rgb_frame, 1)
    cap.release()
    if len(dets) > 0:
        shape = shape_predictor(rgb_frame, dets[0])
        current_embedding = face_rec_model.compute_face_descriptor(rgb_frame, shape)
        current_embedding = np.array(current_embedding)
        max_similarity = 0.8
        recognized_name = None
        for name, embeddings_list in known_people.items():
            for stored_embedding in embeddings_list:
                stored_embedding = np.array(stored_embedding)
                similarity = np.dot(current_embedding, stored_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                )
                if similarity > max_similarity and similarity > similarity_threshold:
                    max_similarity = similarity
                    recognized_name = name
        if recognized_name:
            current_person = recognized_name
            if current_person != previous_person:
                previous_person = current_person
                success_count += 1
                failure_count = 0
                if success_count >= 3:
                    similarity_threshold = min(similarity_threshold + 0.05, 1.0)
                return f"مرحبًا اهلا وسهلا بك، {recognized_name}!"
        else:
            current_person = None
            previous_person = None
            failure_count += 1
            success_count = 0
            if failure_count >= 3:
                similarity_threshold = max(similarity_threshold - 0.05, 0.5)
            return None

# الاستماع إلى المستخدم
def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Press Ctrl+C to stop.")
        while True:
            try:
                audio = recognizer.listen(source)
                user_input = recognizer.recognize_google(audio, language="ar")
                if user_input:
                    return user_input
            except sr.UnknownValueError:
                st.write("لم يتم التعرف على الصوت، الرجاء المحاولة مجددًا.")
            except sr.RequestError as e:
                st.write(f"خطأ في خدمة التعرف الصوتي: {e}")

# التحدث بالنص
def speak_text(text):
    tts = gTTS(text=text, lang='ar')
    filename = "output.mp3"
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()
    os.remove(filename)

# الحصول على الرد
def get_response(user_input):
    cleaned_input = cleaning_text(user_input)
    if re.search(r'\b(اريدك ان تعرفني|هل تعرفني|تعرفني|بتعرفني|تذكرنيمن انا)\b', cleaned_input):
        if current_person in known_people:
            return f"نعم، أعرفك. أنت {current_person}."
        else:
            return "لا، لا أعرفك. هل يمكنك تعريف نفسك؟"
    if cleaned_input in learned_responses:
        return learned_responses[cleaned_input]
    input_tfidf = tfidf.transform([cleaned_input]).toarray()
    prediction = model.predict(input_tfidf)
    tag_index = np.argmax(prediction)
    confidence = prediction[0][tag_index]
    if confidence > 0.5:
        tag = label_encoder.inverse_transform([tag_index])
        return tag[0]
    else:
        return "لم أفهم سؤالك جيدًا، هل يمكنك التوضيح أكثر؟"

# بدء تشغيل البوت
def start_voice_bot():
    global current_person, previous_person, similarity_threshold
    mic = listen_to_user()
    if mic in ['مرحبا', 'السلام عليكم', 'أقصى']:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak_text("تعذر تشغيل الكاميرا.")
            return
        in_chat = False
        while True:
            ret, frame = cap.read()
            if not ret:
                speak_text("فشل في قراءة الإطار من الكاميرا.")
                cap.release()
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = face_detector(rgb_frame, 1)
            if len(dets) > 0:
                shape = shape_predictor(rgb_frame, dets[0])
                current_embedding = face_rec_model.compute_face_descriptor(rgb_frame, shape)
                current_embedding = np.array(current_embedding)
                max_similarity = 0
                recognized_name = None
                for name, embeddings_list in known_people.items():
                    for stored_embedding in embeddings_list:
                        stored_embedding = np.array(stored_embedding)
                        similarity = np.dot(current_embedding, stored_embedding) / (
                            np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                        )
                        if similarity > max_similarity and similarity > similarity_threshold:
                            max_similarity = similarity
                            recognized_name = name
                if recognized_name:
                    if recognized_name != current_person:
                        if in_chat and current_person:
                            speak_text(f"وداعًا {current_person}.")
                            in_chat = False
                        current_person = recognized_name
                        previous_person = current_person
                        speak_text(f"مرحبًا {current_person}! كيف يمكنني مساعدتك؟")
                        in_chat = True
                        chat_with_user(cap)
                else:
                    if not recognized_name:
                        if current_person:
                            speak_text(f"وداعًا {current_person}.")
                            current_person = None
                            previous_person = None
                            in_chat = False
                        speak_text("مرحبًا بك! كيف يمكنني مساعدتك؟")
                        current_person = "زائر"
                        in_chat = True
                        chat_with_user(cap)
                    name_input = listen_to_user()
                    if name_input:
                        clean_name = extract_name(name_input)
                        if clean_name:
                            extract_and_save_face_embedding(clean_name, rgb_frame)
                            current_person = clean_name
                            speak_text(f"مرحبًا {current_person}! تشرفت بمعرفتك.")
                            in_chat = True
                            chat_with_user(cap)
            else:
                if current_person:
                    speak_text(f"وداعًا {current_person}.")
                    current_person = None
                    previous_person = None
                    in_chat = False
            cv2.waitKey(100)
        cap.release()

# الدردشة مع المستخدم
def chat_with_user(cap):
    global current_person
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = face_detector(rgb_frame, 1)
        if len(dets) == 0:
            speak_text("وداعا.")
            current_person = None
            break
        user_input = listen_to_user()
        if user_input.lower() in ["توقف", "انهاء", "إيقاف"]:
            speak_text("تم إنهاء الدردشة.")
            current_person = None
            break
        response = get_response(user_input)
        speak_text(response)
        if "لا أعرفك" in response:
            speak_text("هل تخبرني ما اسمك؟")
            name_input = listen_to_user()
            if name_input:
                clean_name = extract_name(name_input)
                if clean_name:
                    extract_and_save_face_embedding(clean_name, rgb_frame)
                    current_person = clean_name
                    speak_text(f"مرحبًا {current_person}! تشرفت بمعرفتك.")

# تحسين تمثيلات الوجه
def capture_varied_representations():
    global current_person
    if not current_person:
        speak_text("لم يتم التعرف على شخص لإضافة تمثيلات له.")
        return
    speak_text(f"الرجاء تحريك وجهك في زوايا مختلفة لتحسين التعرف على {current_person}.")
    cap = cv2.VideoCapture(0)
    for i in range(3):
        speak_text(f"التقاط الصورة {i + 1}. الرجاء تغيير زاوية وجهك.")
        time.sleep(2)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extract_and_save_face_embedding(current_person, rgb_frame)
    cap.release()
    speak_text(f"تمت إضافة تمثيلات جديدة للشخص {current_person}.")

# الواجهة الرئيسية
def main():
    st.title("Arabic Voice Chatbot")
    st.write("Welcome to the Arabic Voice Chatbot!")

    if st.button("Start Bot"):
        start_voice_bot()

    if st.button("Improve Representation"):
        capture_varied_representations()

if __name__ == "__main__":
    main()