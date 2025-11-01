import speech_recognition as sr

recognizer = sr.Recognizer()

print("Testing microphone... Please speak a phrase when prompted.")

try:
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please be quiet.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Ready! Say something...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        print("Processing speech...")
        
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        print("Microphone test successful!")
except sr.WaitTimeoutError:
    print("Error: No speech detected. Timeout.")
except sr.UnknownValueError:
    print("Error: Could not understand audio. Speech unclear.")
except sr.RequestError:
    print("Error: Could not request results from Google Speech Recognition service.")
except Exception as e:
    print(f"Error: {e}")

print("Press Enter to exit...")
input()
