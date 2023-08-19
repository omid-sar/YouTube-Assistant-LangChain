# Import the necessary packages
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


# Load  OpenAI API key from .env file and set it as the API key
openai_key = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

video_url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
persist_directory = '../../data/processed'
embeddings = OpenAIEmbeddings()


loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
docs = text_splitter.split_documents(transcript)
#docs[0].page_content

vectordb = Chroma.from_documents(documents=docs,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)

def youtube_url_db(video_url):
    # Load the video from the URL
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load())

    # Split the video into sentences
    splitter = RecursiveCharacterTextSplitter()
    sentences = splitter.split_text

    # Embed the sentences
    sentence_embeddings = embeddings.embed(sentences)

    # Store the sentences and their embeddings in a vector store
    vector_store = chroma.VectorStore()
    vector_store.store(sentences, sentence_embeddings)

    # Get the most similar sentences to the query
    query = "What is the most important thing in life?"
    results = vector_store.most_similar(query, topn=5)
    return results
