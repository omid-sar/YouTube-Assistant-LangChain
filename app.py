# Import the necessary packages
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import textwrap
import streamlit as st
from apikey import apikey

# Load  OpenAI API key and define directory to store the data
os.environ["OPENAI_API_KEY"] = apikey
persist_directory = "../../data/processed"

# Load the OpenAI Embeddings, LLM , PromptTemplate and LLMChain
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)
# Define the template for the prompt
template = """You can provide answers about YouTube videos using their transcripts.

    For the question: {question}
    Please refer to the video transcript: {docs_page_content}

    Rely solely on the transcript's factual data to respond.

    If the information isn't sufficient, simply state "I don't know".

    Ensure your answers are comprehensive and in-depth.
    """
prompt = PromptTemplate(
    input_variables=["question", "docs_page_content"],
    template=template,
)
chain = LLMChain(llm=llm, prompt=prompt)


# Setup streamlit
st.title("YouTube Video Transcript Analyzer")
# *** YOUR VIDEO URL and QUESTION ***
video_url = st.text_input("Enter the YouTube video URL:")
question = st.text_input("Enter your question about the video:")
# add submit button
# submit = st.button("Submit")
#
if video_url and question:
    # load the video transcript
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    # show the video title and author
    info = loader._get_video_info()
    st.write("**Title:**", info["title"])
    st.write("**Author:**", info["author"])
    # Split the transcript into chunks with 1500 characters and 150 characters overlap
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = text_splitter.split_documents(transcript)
    # docs[0].page_content
    # Create the vector database which will be used to search for similar sentences
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )

    # Search for the most similar sentences to the question and concatenate top 3 vectors
    docs = vectordb.similarity_search(query=question, k=3)
    docs_page_content = " ".join([doc.page_content for doc in docs])
    # docs[0].page_content
    # send the question and the top 3 sentences to the LLMChain and print the response
    response = chain.run(question=question, docs_page_content=docs_page_content)
    st.write(textwrap.fill(response, width=85))
