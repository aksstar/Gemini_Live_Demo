# Gemini Live Audio Chat

This project is a live demonstration of a real-time, two-way audio chat with the Gemini AI model. It uses the Gemini API's live functionality to stream audio to and from the model, creating a seamless conversational experience.

## Screenshot

![Screenshot](Screenshot%202026-01-20%20at%2012.02.31%E2%80%AFPM.png)

## Features

*   **Real-time Audio Streaming:** Engages in a natural, real-time conversation with Gemini.
*   **Live Transcription:** Displays live transcriptions of both your speech and Gemini's responses.
*   **Simple Web Interface:** An intuitive and easy-to-use interface built with Gradio.
*   **Asynchronous Handling:** Built with Python's `asyncio` to handle concurrent audio streaming and API interactions efficiently.

## How It Works

The application captures audio from your microphone using `pyaudio` and streams it directly to the Gemini API. In response, Gemini streams audio back, which is then played through your speakers. The application also receives and displays live transcriptions of the conversation.

The frontend is a simple Gradio interface with buttons to start and stop the session and text boxes to display the conversation.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/google/generative-ai-docs
    cd generative-ai-docs/demos/gemini-live
    ```

2.  **Install the dependencies:**
    Make sure you have Python 3.9+ installed. Then, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: On some systems, you may need to install `portaudio` for `pyaudio` to work correctly. For example, on Debian/Ubuntu:*
    ```bash
    sudo apt-get install portaudio19-dev
    ```

3.  **Configure the application:**
    Open `app.py` and set your `PROJECT_ID` and `LOCATION` at the top of the file.

4.  **Run the application:**
    ```bash
    python app.py
    ```

    This will start a local web server, and you can access the application by navigating to the provided URL in your web browser.

## Usage

1.  Click the "Start Session" button.
2.  Allow the browser to access your microphone.
3.  Start speaking, and you should hear Gemini's response.
4.  Click the "Stop Session" button to end the conversation.