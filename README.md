
# YouTube Video Transcript Analyzer

## Goal

The goal of this application is to analyze YouTube video transcripts. It takes a YouTube video URL and a user-input question, processes the video transcript, performs a similarity search to find the most related chunk of the transcript, and then sends it to OpenAI to generate a comprehensive response based on the provided question.

## Overview

The application uses the `langchain` library to handle various tasks such as loading YouTube video transcripts, converting transcripts to embeddings, performing similarity searches, and integrating with OpenAI's language models. The application is built using Streamlit, which provides an interactive user interface for inputting the video URL and the question.

## Features

- **YouTube Video Transcript Loading**: Load and process YouTube video transcripts using `YoutubeLoader` from the `langchain` library.
- **Text Splitting**: Split the transcript into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Embeddings Conversion**: Convert video transcripts to embeddings using `OpenAIEmbeddings`.
- **Similarity Search**: Perform a similarity search to find the most relevant section of the transcript using `Chroma`.
- **OpenAI Integration**: Utilize OpenAI's language models to generate responses based on the processed transcript and user-input question.

## Prerequisites

- Python 3.x
- Streamlit
- OpenAI API Key

## Setup and Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/omid-sar/YouTube-Assistant-LangChain.git
   cd YouTube-Assistant-LangChain
   ```

2. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```sh
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. Open the app in your web browser and input the YouTube video URL and your question about the video.

3. The app will process the video transcript, perform a similarity search, and use OpenAI to generate a response, which will be displayed on the screen.


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for details.
```

