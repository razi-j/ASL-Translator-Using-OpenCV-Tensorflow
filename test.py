import pyttsx3
def tts(sentence):
    tts = pyttsx3.init()
    #tts.connect('started-utterance', main)
    tts.setProperty('rate', 130)
    tts.say(sentence)
    tts.runAndWait()

tts("Hello")