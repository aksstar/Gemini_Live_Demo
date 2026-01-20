
import asyncio
import threading
import traceback
import queue
import time
import gradio as gr
print(f"Gradio version: {gr.__version__}")
from google import genai
import pyaudio

# --- Gemini API config ---
PROJECT_ID = "aakash-test-env"
LOCATION = "us-central1"

# --- pyaudio config ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# --- Live API config ---
MODEL = "gemini-live-2.5-flash-native-audio"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful and friendly AI assistant.",
    "input_audio_transcription": {},
    "output_audio_transcription": {},
}

class AudioChatSession:
    def __init__(self):
        self._client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        self._pya = pyaudio.PyAudio()
        self._audio_queue_output = asyncio.Queue()
        self._audio_queue_mic = asyncio.Queue(maxsize=5)
        self._input_transcript_queue = queue.Queue()
        self._output_transcript_queue = queue.Queue()
        self._audio_stream = None
        self._run_task = None
        self._loop = None
        self._is_running = False

    async def _listen_audio(self):
        """Listens for audio and puts it into the mic audio queue."""
        mic_info = self._pya.get_default_input_device_info()
        self._audio_stream = await asyncio.to_thread(
            self._pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while self._is_running:
            try:
                data = await asyncio.to_thread(self._audio_stream.read, CHUNK_SIZE, **kwargs)
                await self._audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})
            except asyncio.CancelledError:
                break

    async def _send_realtime(self, session):
        """Sends audio from the mic audio queue to the GenAI session."""
        while self._is_running:
            try:
                msg = await asyncio.wait_for(self._audio_queue_mic.get(), timeout=1.0)
                await session.send_realtime_input(audio=msg)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _receive_audio(self, session):
        """Receives responses from GenAI and puts audio data into the speaker audio queue."""
        turn = session.receive()
        while self._is_running:
            try:
                response = await asyncio.wait_for(turn.__anext__(), timeout=1.0)
                if response.server_content:
                    if response.server_content.interrupted:
                        # If the server reports an interruption, clear the audio queue to stop playback
                        while not self._audio_queue_output.empty():
                            self._audio_queue_output.get_nowait()
                        print("Audio playback interrupted by server.")
                        continue

                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                self._audio_queue_output.put_nowait(part.inline_data.data)
                    if response.server_content.input_transcription and response.server_content.input_transcription.text:
                        self._input_transcript_queue.put(response.server_content.input_transcription.text)
                    if response.server_content.output_transcription and response.server_content.output_transcription.text:
                        self._output_transcript_queue.put(response.server_content.output_transcription.text)

            except asyncio.TimeoutError:
                continue
            except StopAsyncIteration:
                # This just means the current async generator from session.receive() is done.
                # Get the next one.
                turn = session.receive()
            except asyncio.CancelledError:
                break


    async def _play_audio(self):
        """Plays audio from the speaker audio queue."""
        stream = await asyncio.to_thread(
            self._pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while self._is_running:
            try:
                bytestream = await asyncio.wait_for(self._audio_queue_output.get(), timeout=1.0)
                await asyncio.to_thread(stream.write, bytestream)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
        stream.close()

    async def _run(self):
        """Main function to run the audio loop."""
        self._is_running = True
        try:
            async with self._client.aio.live.connect(
                model=MODEL, config=CONFIG
            ) as live_session:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._send_realtime(live_session))
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio(live_session))
                    tg.create_task(self._play_audio())
        except asyncio.CancelledError:
            print("Session cancelled.")
        except Exception as e:
            print(f"An error occurred: {e}")
            if isinstance(e, ExceptionGroup):
                for i, exc in enumerate(e.exceptions):
                    print(f"  Sub-exception {i+1}: {exc}")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
            else:
                traceback.print_exc()
        finally:
            self._is_running = False
            if self._audio_stream:
                self._audio_stream.close()
                self._audio_stream = None
            # The PyAudio object is terminated in the stop method

    def start(self):
        if self._is_running:
            return "Session is already running."

        def loop_in_thread():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._run_task = self._loop.create_task(self._run())
            try:
                self._loop.run_until_complete(self._run_task)
            finally:
                self._loop.close()

        threading.Thread(target=loop_in_thread, daemon=True).start()
        return "Session started. Speak into the microphone."

    def stop(self):
        if not self._is_running or not self._run_task or not self._loop:
            return "Session is not running."

        self._is_running = False
        self._loop.call_soon_threadsafe(self._run_task.cancel)
        
        # Cleanup PyAudio
        self._pya.terminate()

        # Re-initialize for next session
        self._pya = pyaudio.PyAudio()
        self._audio_queue_output = asyncio.Queue()
        self._audio_queue_mic = asyncio.Queue(maxsize=5)
        self._input_transcript_queue = queue.Queue()
        self._output_transcript_queue = queue.Queue()

        return "Session stopped."


chat_session = AudioChatSession()

with gr.Blocks() as demo:
    gr.Markdown("<h1>Gemini Live Audio Chat</h1>")
    gr.Markdown("Start a real-time voice conversation with Gemini. Press 'Start' and begin speaking.")
    
    with gr.Row():
        start_button = gr.Button("Start Session")
        stop_button = gr.Button("Stop Session")

    status_display = gr.Label(value="Status: Not Connected")
    
    with gr.Row():
        input_transcript_box = gr.Textbox(label="Your Speech", interactive=False, lines=5)
        output_transcript_box = gr.Textbox(label="Gemini's Response", interactive=False, lines=5)

    def start_and_update_transcripts():
        status_update = "Status: Connected. Speak now!"
        
        # Initial yield with basic values
        yield status_update, gr.Button(interactive=False), gr.Button(interactive=True), "", ""

        chat_session.start()
        
        input_full_text = ""
        output_full_text = ""
        
        while chat_session._is_running:
            new_input_text = ""
            while not chat_session._input_transcript_queue.empty():
                new_input_text += chat_session._input_transcript_queue.get_nowait() + " "
            
            new_output_text = ""
            while not chat_session._output_transcript_queue.empty():
                new_output_text += chat_session._output_transcript_queue.get_nowait() + " "

            if new_input_text:
                input_full_text += new_input_text
            if new_output_text:
                output_full_text += new_output_text
            
            # Yield updates without fancy styling or variant changes
            yield status_update, gr.Button(interactive=False), gr.Button(interactive=True), input_full_text, output_full_text
            time.sleep(0.2)
        
        # Final update after stopping
        yield "Status: Session Ended", gr.Button(interactive=True), gr.Button(interactive=False), input_full_text, output_full_text

    def stop_wrapper():
        chat_session.stop()
        # Return basic values for buttons
        return "Status: Session Stopped", gr.Button(interactive=True), gr.Button(interactive=False)

    start_button.click(
        start_and_update_transcripts, 
        outputs=[status_display, start_button, stop_button, input_transcript_box, output_transcript_box]
    )
    stop_button.click(
        stop_wrapper, 
        outputs=[status_display, start_button, stop_button]
    )

if __name__ == "__main__":
    demo.launch()
