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


# Load  OpenAI API key from .env file and set it as the API key
openai_key = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

# *** YOUR VIDEO URL ***
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


# *** YOUR QUESTION ABUT THE VIDEO ***
question = "what did course instructor say about computer learning?"

docs = vectordb.similarity_search(query=question, k=3)
docs_page_content = " ".join([doc.page_content for doc in docs])
docs[0].page_content


template = """ You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Please address the question: {question}
        By referencing the provided video transcript: {docs_page_content}   
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed."""

prompt = PromptTemplate(
    input_variables=["question", "docs_page_content"],
    template=template,
)

llm = OpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(question=question, docs_page_content=docs_page_content)
print(textwrap.fill(response, width=85))
