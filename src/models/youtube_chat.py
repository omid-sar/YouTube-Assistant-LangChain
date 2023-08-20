# Import the necessary packages
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# Load  OpenAI API key from .env file and set it as the API key
openai_key = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

video_url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
persist_directory = "../../data/processed"
embeddings = OpenAIEmbeddings()


loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
docs = text_splitter.split_documents(transcript)
# docs[0].page_content

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)


# Create a database of the youtube video url
def youtube_url_db(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(transcript)
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )
    return vectordb


question = "What is course about?"

docs = vectordb.similarity_search(query=question, k=3)
docs_page_content = " ".join([doc.page_content for doc in docs])

llm = OpenAI(temperature=0)
template = """
You are a knowledgeable assistant proficient in answering queries related
to YouTube videos by analyzing their transcripts.

Please address the query: {question}
By referencing the provided video transcript: {docs_page_content}

If the answer isn't apparent in the transcript, kindly mention that you're unsure.
Please refrain from speculating. Conclude all responses with "Thanks for asking!"""
prompt = PromptTemplate(
    input_variables=["question", "docs_page_content"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(question=question, docs_page_content=docs_page_content)
print(response)
