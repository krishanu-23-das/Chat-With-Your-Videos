import streamlit as st
from pytube import YouTube
from dotenv import load_dotenv
import os
import whisper
from moviepy.editor import *
import datetime as dt
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

def load_video(url):
    yt = YouTube(url)
    #video_stream = yt.streams.filter(file_extension="mp4")
    target_dir = os.path.join('C:\\Users\\daskr\\PycharmProjects\\Chat_with your_Youtube_videos', 'Youtube')

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    #yt.streams.first().download()(output_path=target_dir)
    print('----DOWNLOADING VIDEO FILE----')
    file_path = yt.streams.filter(only_audio=True, subtype='webm', abr='160kbps').first().download(output_path=target_dir)
    return file_path


def process_video(path):
    file_dir = path
    print('Transcribing Video with whisper base model')
    model = whisper.load_model("base")
    result = model.transcribe(file_dir)
    return result

def process_text(result):
    texts, start_time_list = [], []
    for res in result['segments']:
        start = res['start']
        text = res['text']

        start_time = dt.datetime.fromtimestamp(start)
        start_time_formatted = start_time.strftime("%H:%M:%S")

        #creating list of texts and start_time
        texts.append(''.join(text))
        start_time_list.append(start_time_formatted)

    texts_with_timestamps = dict(zip(texts, start_time_list))

    formatted_texts = {
        text: dt.datetime.strptime(str(timestamp), '%H:%M:%S')
        for text, timestamp in texts_with_timestamps.items()
    }

    #grouping the sentences in the interval of 30 seconds, & stoding the texts and starting time
    # in group_texts & time_list reps

    grouped_texts = []
    current_group = ''
    time_list = [list(formatted_texts.values())[0]]
    previous_time = None
    time_difference = dt.timedelta(seconds=30)

    # Group texts based on time difference
    for text, timestamp in formatted_texts.items():

        if previous_time is None or timestamp - previous_time <= time_difference:
            current_group += text
        else:
            grouped_texts.append(current_group)
            time_list.append(timestamp)
            current_group = text
        previous_time = time_list[-1]

    # Append the last group of texts
    if current_group:
        grouped_texts.append(current_group)

    return grouped_texts, time_list


def get_vectorstore(grouped_texts, time_list):

    text = grouped_texts
    time_stamps = time_list

    time_stamps = [{'source': str(t.time())} for t in time_stamps]
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorestore = FAISS.from_texts(texts=text, embeddings=embeddings,
                                    collection_name='test', metadatas=time_stamps)
    return vectorestore

def get_conversation(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    favicon_url = 'https://th.bing.com/th/id/R.087b4dc55ac459f86e6d11d402095394?rik=SfrwQLE7z60OLg&pid=ImgRaw&r=0&sres=1&sresct=1'

    st.set_page_config(page_title='Chat with YouTube videos', page_icon=favicon_url)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    st.header('Chat with your videos :film_frames:')
    user_question = st.text_input('Enter your query here')

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader('Your Video file')
        url = st.text_input('Enter your URL here and click on "Process"')
        if st.button('Process'):
            with st.spinner('Processing'):
                st.video(url)

                #load the video
                path = load_video(url)

                #convert audio to text file using whisper
                result = process_video(path)

                #Embeed & transfer the converted text into vectorstore
                grouped_texts, time_list = process_text(result)
                vectorstore = get_vectorstore(grouped_texts, time_list)

                # creating conversational chatbot
                st.session_state.conversation = get_conversation(vectorstore)
                st.write("____MODEL____LOADED______")
                st.write("NOW YOU CAN ASK YOUR QUESTIONS.")




if __name__ == '__main__':
    main()