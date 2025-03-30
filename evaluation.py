import wave
from groq import Groq
import pyperclip
import edge_tts  # Import the Edge TTS library
from pydub import AudioSegment
import pyaudio
import asyncio
import threading
import os

groq_client = Groq(api_key="put your groq api here")

sys_msg = (
    'You are an AI voice assistant. Your name is "Neha". '
    'Generate the most useful and factual response possible, '
    'carefully considering all previous generated text in your response before '
    'adding new tokens to the response. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

def groq_prompt(prompt):
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content '
        'or calling no functions is best for a voice assistant to respond '
        'to the users prompt. You will respond with only one selection from this list: '
        '["extract clipboard", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]

    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

async def read_response_aloud(text):
    communicate = edge_tts.Communicate(text, "en-US-AvaNeural")
    await communicate.save("response.mp3")

    # Convert MP3 to WAV
    mp3_audio = AudioSegment.from_mp3("response.mp3")
    wav_path = "response.wav"
    mp3_audio.export(wav_path, format="wav")

    # Play the WAV file using wave module
    CHUNK = 1024
    wf = wave.open(wav_path, 'rb')
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read data
    data = wf.readframes(CHUNK)

    # Play stream
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    # Stop stream
    stream.stop_stream()
    stream.close()

    # Close PyAudio
    p.terminate()

def read_response_and_play(text):
    asyncio.run(read_response_aloud(text))

while True:
    prompt = input('USER: ')
    
    call = function_call(prompt)

    if 'extract clipboard' in call:
        print('Copying clipboard text')
        paste = get_clipboard_text()
        prompt = f'{prompt}\n\n CLIPBOARD CONTENT: {paste}'
    
    response = groq_prompt(prompt=prompt)
    print(response)

    # Start the text-to-speech in a separate thread
    tts_thread = threading.Thread(target=read_response_and_play, args=(response,))
    tts_thread.start()