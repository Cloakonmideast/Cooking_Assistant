import speech_recognition as sr
import pyttsx3
import time
import threading
import queue
import json
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class VoiceCookingAssistant:
    def __init__(self, trigger_word="assistant"):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.trigger_word = trigger_word.lower()

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

        self.listening = True
        self.waiting_for_command = False
        self.command_start_time = 0
        self.is_cooking = False
        self.current_recipe = None
        self.current_step = 0

        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._process_tts_queue, daemon=True)
        self.tts_thread.start()

        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
        self.engine.setProperty('rate', 150)

        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def _process_tts_queue(self):
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
            self.tts_queue.task_done()

    def speak(self, text):
        print(f"Assistant: {text}")
        self.tts_queue.put(text)

    def listen(self):
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("🎤 Listening for 'assistant'...")

            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{text}'")

                    if self.waiting_for_command:
                        print(f"🎯 Processing command: {text}")
                        self.process_command(text)
                        self.waiting_for_command = False
                        print("🎤 Back to listening for 'assistant'...")

                    elif self.trigger_word in text:
                        print("✅ Trigger word detected!")
                        self.speak("Yes, I'm listening. What can I help you with?")
                        self.waiting_for_command = True
                        self.command_start_time = time.time()
                        print("⏰ Waiting for your command...")

                    if self.waiting_for_command and (time.time() - self.command_start_time > 10):
                        print("⏰ Command timeout")
                        self.speak("I didn't hear a command. Say 'assistant' again when you're ready.")
                        self.waiting_for_command = False

                except sr.WaitTimeoutError:
                    if self.waiting_for_command and (time.time() - self.command_start_time > 10):
                        print("⏰ Command timeout during silence")
                        self.speak("I didn't hear anything. Say 'assistant' again when you're ready.")
                        self.waiting_for_command = False
                    else:
                        print(".", end="", flush=True)

                except sr.UnknownValueError:
                    print("?", end="", flush=True)

                except sr.RequestError as e:
                    print(f"\n🌐 Network error: {e}")
                    time.sleep(2)

                except Exception as e:
                    print(f"\n❌ Listening error: {e}")
                    time.sleep(1)

    def get_recipe_from_gemini(self, query):
        try:
            print(f"🔍 Generating recipe for: {query}")
            prompt = (
                f"Give me a simple recipe for {query}. "
                "Return in JSON format like this:\n"
                "{\n"
                "  \"title\": \"Recipe Title\",\n"
                "  \"ingredients\": [\"item1\", \"item2\"],\n"
                "  \"steps\": [\"step1\", \"step2\"]\n"
                "}"
            )

            response = self.model.generate_content(prompt)
            text_response = response.text.strip()

            json_start = text_response.find('{')
            json_end = text_response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text_response[json_start:json_end]
                try:
                    recipe_data = json.loads(json_str)
                    print("✅ Recipe parsed successfully!")
                    return recipe_data
                except json.JSONDecodeError:
                    print("❌ JSON parsing failed")
            else:
                print("❌ No JSON found in response")

        except Exception as e:
            print(f"❌ Error fetching recipe: {e}")

        return None

    def process_command(self, command):
        print(f"🔄 Processing: '{command}'")

        if "find recipe for" in command or "how to make" in command or "recipe" in command:
            query = command.replace("find recipe for", "").replace("how to make", "").replace("recipe for", "").replace("recipe", "").strip()

            if not query:
                self.speak("What would you like a recipe for?")
                return

            threading.Thread(target=self.fetch_and_announce_recipe, args=(query,), daemon=True).start()

        elif "start cooking" in command and self.current_recipe:
            self.is_cooking = True  # ✅ FIXED: mark cooking started
            self.speak("Great! Let's start cooking. Here's the first step.")
            self.read_current_step()

        elif "next step" in command and self.is_cooking:
            self.current_step += 1
            if self.current_step < len(self.current_recipe['steps']):
                self.read_current_step()
            else:
                self.speak("Congratulations! That was the last step. Your dish should be ready. Enjoy your meal!")
                self.is_cooking = False

        elif "repeat" in command and self.is_cooking:
            self.speak("Let me repeat the current step.")
            self.read_current_step()

        elif "previous step" in command and self.is_cooking:
            if self.current_step > 0:
                self.current_step -= 1
                self.speak("Going back to the previous step.")
                self.read_current_step()
            else:
                self.speak("You're already at the first step.")

        elif "ingredients" in command and self.current_recipe:
            self.speak("Here are the ingredients again:")
            for i, ingredient in enumerate(self.current_recipe['ingredients'], 1):
                self.speak(f"{i}. {ingredient}")
                time.sleep(0.3)

        elif "stop" in command or "exit" in command or "quit" in command:
            self.speak("Goodbye! Happy cooking!")
            self.listening = False

        else:
            self.speak("I didn't understand that command. Try saying 'find recipe for' followed by a dish name, or say 'stop' to exit.")

    def fetch_and_announce_recipe(self, query):
        self.speak(f"Looking for a recipe for {query}. This might take a moment.")
        recipe = self.get_recipe_from_gemini(query)

        if recipe:
            self.current_recipe = recipe
            self.current_step = 0
            self.is_cooking = False

            self.speak(f"I found a recipe for {recipe['title']}. Here are the ingredients:")
            for i, ingredient in enumerate(recipe['ingredients'], 1):
                self.speak(f"{i}. {ingredient}")
                time.sleep(0.3)

            self.speak("Say 'assistant' then 'start cooking' when you're ready to begin.")
        else:
            self.speak("I couldn't find a recipe for that. Please try a different dish or check your internet connection.")

    def read_current_step(self):
        if self.current_recipe and 0 <= self.current_step < len(self.current_recipe['steps']):
            print(f"📖 Reading step {self.current_step + 1}")
            step = self.current_recipe['steps'][self.current_step]
            step_number = self.current_step + 1
            total_steps = len(self.current_recipe['steps'])
            self.speak(f"Step {step_number} of {total_steps}: {step}")
            if step_number < total_steps:
                self.speak("Say 'assistant' then 'next step' when you're ready to continue.")

    def run(self):
        print("🍳 Voice Cooking Assistant Starting...")
        print("=" * 50)
        self.speak("Voice Cooking Assistant is ready!")
        self.speak("Say 'assistant' followed by your command.")
        time.sleep(4)

        listen_thread = threading.Thread(target=self.listen, daemon=True)
        listen_thread.start()

        try:
            while self.listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.listening = False
            self.tts_queue.put(None)
            self.speak("Goodbye!")

if __name__ == "__main__":
    assistant = VoiceCookingAssistant(trigger_word="assistant")
    assistant.run()
