import speech_recognition as sr
import pyttsx3
import time
import threading
import queue
import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VoiceCookingAssistant:
    def __init__(self, trigger_word="assistant"):
        # Initialize speech recognition and text-to-speech engines
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.trigger_word = trigger_word.lower()
        
        # Set up Gemini API key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        # Control variables
        self.listening = True
        self.is_cooking = False
        self.current_recipe = None
        self.current_step = 0
        
        # Command queue for handling interruptions
        self.command_queue = queue.Queue()
        
        # Set voice properties
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Female voice
        self.engine.setProperty('rate', 150)  # Speed of speech
    
    def speak(self, text):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
    
    def listen(self):
        """Continuously listen for voice commands"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"User: {text}")
                    
                    # Check for trigger word
                    if self.trigger_word in text:
                        self.speak("Yes, I'm listening")
                        self.command_queue.put(("interrupt", None))
                        
                        # Listen for the command after trigger
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                        command = self.recognizer.recognize_google(audio).lower()
                        print(f"Command: {command}")
                        self.command_queue.put(("command", command))
                    
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"Error: {e}")
    
    def get_recipe_from_gemini(self, query):
        """Get recipe information from Gemini API"""
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Give me a recipe for {query}. Return in JSON format with the following structure: {{\"title\": \"Recipe Title\", \"ingredients\": [\"item1\", \"item2\"], \"steps\": [\"step1\", \"step2\"]}}"
                }]
            }]
        }
        
        try:
            response = requests.post(self.gemini_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                text_response = result['candidates'][0]['content']['parts'][0]['text']
                
                # Extract JSON from the response
                json_start = text_response.find('{')
                json_end = text_response.rfind('}') + 1
                json_str = text_response[json_start:json_end]
                
                recipe_data = json.loads(json_str)
                return recipe_data
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching recipe: {e}")
            return None
    
    def process_command(self, command):
        """Process voice commands"""
        if "find recipe for" in command or "how to make" in command or "recipe" in command:
            query = command.replace("find recipe for", "").replace("how to make", "").replace("recipe", "").strip()
            self.speak(f"Looking for a recipe for {query}")
            recipe = self.get_recipe_from_gemini(query)
            
            if recipe:
                self.current_recipe = recipe
                self.current_step = 0
                self.is_cooking = True
                
                self.speak(f"I found a recipe for {recipe['title']}. Here are the ingredients:")
                for ingredient in recipe['ingredients']:
                    self.speak(ingredient)
                    time.sleep(0.5)
                
                self.speak("Let me know when you're ready to start cooking.")
            else:
                self.speak("I couldn't find a recipe for that. Please try again.")
        
        elif "start cooking" in command and self.current_recipe:
            self.speak("Let's start cooking. I'll guide you through each step.")
            self.read_current_step()
        
        elif "next step" in command and self.is_cooking:
            self.current_step += 1
            if self.current_step < len(self.current_recipe['steps']):
                self.read_current_step()
            else:
                self.speak("That was the last step. Enjoy your meal!")
                self.is_cooking = False
        
        elif "repeat" in command and self.is_cooking:
            self.read_current_step()
        
        elif "previous step" in command and self.is_cooking:
            self.current_step = max(0, self.current_step - 1)
            self.read_current_step()
        
        elif "ingredients" in command and self.current_recipe:
            self.speak("Here are the ingredients again:")
            for ingredient in self.current_recipe['ingredients']:
                self.speak(ingredient)
                time.sleep(0.5)
        
        elif "stop" in command or "exit" in command or "quit" in command:
            self.speak("Stopping the cooking assistant. Goodbye!")
            self.listening = False
        
        else:
            self.speak("I didn't understand that command. Please try again.")
    
    def read_current_step(self):
        """Read the current recipe step"""
        if self.current_recipe and 0 <= self.current_step < len(self.current_recipe['steps']):
            step = self.current_recipe['steps'][self.current_step]
            self.speak(f"Step {self.current_step + 1}: {step}")
    
    def run(self):
        """Run the cooking assistant"""
        self.speak("Voice Cooking Assistant is ready. Say 'assistant' followed by your command.")
        
        # Start listening thread
        listen_thread = threading.Thread(target=self.listen)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Main processing loop
        try:
            while self.listening:
                try:
                    action, command = self.command_queue.get(timeout=1)
                    
                    if action == "interrupt":
                        # Stop any ongoing speech
                        self.engine.stop()
                    elif action == "command":
                        self.process_command(command)
                    
                    self.command_queue.task_done()
                except queue.Empty:
                    pass
                
        except KeyboardInterrupt:
            self.listening = False
            print("Shutting down...")

if __name__ == "__main__":
    assistant = VoiceCookingAssistant(trigger_word="assistant")
    assistant.run()