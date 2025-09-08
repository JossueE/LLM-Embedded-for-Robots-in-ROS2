import os
#THIS IS THE CONFIG FILE
#Here you can control everything of the LLM Package

"""Global"""
LANGUAGE = "es" #The code is actually not prepared to work with other languages, but for future improvements

"""Audio Listener is the node to hear something from the MIC"""
AUDIO_LISTENER_DEVICE_ID = -1 #This is the default input 
AUDIO_LISTENER_CHANNELS = 1 # "mono" or "stereo"
AUDIO_LISTENER_SAMPLE_RATE = 16000
AUDIO_LISTENER_FRAMES_PER_BUFFER = 1000

"""LLM node"""
CONTEXT_LLM = 1024 #The size of the context that your model is going to recieve
THREADS_LLM = os.cpu_count() or 8 #Threads that has available your model 
N_BACH_LLM = 512 #The size of the info that gpu or cpu is going to process
GPU_LAYERS_LLM = 0 #How many layers your model is going to use in GPU, for CPU use "0"
MAX_MOVE_DISTANCE_LLM = 10.0 #Max distance in meters of the robot movement
CHAT_FORMAT_LLM = "chatml-function-calling" #NOT recomended to change unless you change the model


#The model that you want to implement
DEFAULT_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf?download=true"

"""Information - data"""
FUZZY_LOGIC_ACCURACY_KB = 0.75 
FUZZY_LOGIC_ACCURACY_POSE = 0.70
PATH_KB = "src/LLM/data/kb.json"
PATH_POSES = "src/LLM/data/poses.json"

"""Audio Publisher Node"""
AUDIO_PUBLISHER_DEVICE_ID = -1 #This is the default output
AUDIO_PUBLISHER_CHANELS = 1 # "mono" or "stereo"
AUDIO_PUBLISHER_FRAMES_PER_BUFFER = 256
AUDIO_PUBLISHER_DEBUG = True #Show in terminal the debug process

"""Text-to-Speech Node"""
SAMPLE_RATE_TTS = 24000
DEVICE_SELECTOR_TTS = "cpu" # "cpu" or "cuda"
SPEAKER_TTS = "v3_es" #Check other models in Silero TTS
VOICE_TTS = "es_2" #voices are: es_0, es_1, es_2

"""Speech-to-Text Node"""
SAMPLE_RATE_STT = 16000 #Silero works at this sample_rate doesn't change unless it is necesarry
CHANNELS_INPUT_STT = 1 #mono or stereo, silero use mono
DEVICE_SELECTOR_STT = "cpu" # "cpu" or "cuda"
#IMPORTANT the system is prepare to work without this variable, but we have it for noisy enviroments, as a protection method
LISTEN_SECONDS_STT = 10.0 #The time of the phrase that the tts is going to be active after de wake_word detection

"""Wake-Word Node"""
DEFAULT_MODEL_FILENAME_WAKE_WORD= "vosk-model-small-es-0.42" #The model
DEFAULT_MODEL_URL_WAKE_WORD = "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip" #The URL
ACTIVATION_PHRASE_WAKE_WORD = "ok robot" #The Activation Word that the model is going to detect
VARIANTS_WAKE_WORD =  ["ok robot", "okay robot", "hey robot"] #variatons