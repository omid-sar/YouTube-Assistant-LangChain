import os
import streamlit as st
import textwrap
# Import other necessary modules and functions here

# Load OpenAI API key from .env file and set it as the API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

# Set up Streamlit interface
st.title('YouTube Video Transcript Analyzer')

# Input for video URL
video_url = st.text_input('Enter the YouTube video URL:', 'https://www.youtube.com/watch?v=jGwO_UgTS7I')

# Input for question about the video
question = st.text_input('Enter your question about the video:', 'What did course instructor say about computer learning?')

# Button to run the analysis
if st.button('Analyze'):
    # Load and process the video transcript
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = text_splitter.split_documents(transcript)

    # Compute similarity and get relevant parts of the transcript
    vectordb = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    docs = vectordb.similarity_search(query=question, k=3)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    # Generate and display the response
    template = """You can provide answers about YouTube videos using their transcripts.

    For the question: {question}
    Please refer to the video transcript: {docs_page_content}

    Rely solely on the transcript's factual data to respond.

    If the information isn't sufficient, simply state "I don't know".

    Ensure your answers are comprehensive and in-depth.
    """
    prompt = PromptTemplate(input_variables=["question", "docs_page_content"], template=template)
    chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    response = chain.run(question=question, docs_page_content=docs_page_content)
    st.text_area('Response:', textwrap.fill(response, width=85))
