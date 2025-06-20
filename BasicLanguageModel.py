from collections import defaultdict
import random
import math
import difflib
import json
import os

# Suppose you have 10 possible outputs (realistic responses for testing)
possible_outputs = [
    "Hello! How can I help you?",
    "I'm not sure I understand.",
    "Can you tell me more?",
    "Goodbye!",
    "That's interesting!",
    "Why do you say that?",
    "Let's talk about something else.",
    "How are you feeling today?",
    "What would you like to discuss?",
    "I'm here to listen.",
    "You're welcome!"
]

# For each input, store a dict of output: [total_score, count]
ratings = defaultdict(lambda: defaultdict(lambda: [0.5, 1]))

# Example training data: (input, response, rating)
training_data = [
    ("hello", "Hello! How can I help you?", 10),
    ("hello", "I'm not sure I understand.", 2),
    ("hello", "Can you tell me more?", 8),
    ("hello there", "Hello! How can I help you?", 9),
    ("hi", "Hello! How can I help you?", 9),
    ("hi", "What would you like to discuss?", 8),
    ("how are you?", "How are you feeling today?", 10),
    ("how are you?", "I'm not sure I understand.", 3),
    ("how are you doing?", "How are you feeling today?", 9),
    ("how are you doing?", "That's interesting!", 2),
    ("bye", "Goodbye!", 10),
    ("bye", "Let's talk about something else.", 1),
    ("goodbye", "Goodbye!", 9),
    ("goodbye", "I'm here to listen.", 2),
    ("what's up?", "That's interesting!", 8),
    ("what's up?", "Can you tell me more?", 2),
    ("tell me something", "I'm here to listen.", 9),
    ("tell me something", "Let's talk about something else.", 2),
    ("help", "Hello! How can I help you?", 10),
    ("help", "Can you tell me more?", 8),
    ("what do you do?", "I'm here to listen.", 8),
    ("what do you do?", "That's interesting!", 2),
    ("what do you do?", "Let's talk about something else.", 3),
    ("how old are you?", "Why do you say that?", 8),
    ("how old are you?", "I'm not sure I understand.", 1),
    ("what is your name?", "I'm here to listen.", 9),
    ("what is your name?", "Hello! How can I help you?", 8),
    ("can you help me?", "Hello! How can I help you?", 10),
    ("can you help me?", "Can you tell me more?", 9),
    ("can you help me?", "What would you like to discuss?", 8),
    ("thank you", "That's interesting!", 2),
    ("thank you", "Goodbye!", 8),
    ("thanks", "Goodbye!", 9),
    ("thanks", "That's interesting!", 1),
]

pretraining_data = [
    # Greetings
    ("hello", "Hello! How can I help you?", 10),
    ("hi", "Hello! How can I help you?", 10),
    ("hey", "Hello! How can I help you?", 9),
    ("good morning", "Hello! How can I help you?", 8),
    ("good evening", "Hello! How can I help you?", 8),
    ("hello", "Goodbye!", 1),
    # Farewells
    ("bye", "Goodbye!", 10),
    ("goodbye", "Goodbye!", 10),
    ("see you", "Goodbye!", 9),
    ("bye", "Hello! How can I help you?", 1),
    # Asking for help
    ("help", "How are you feeling today?", 7),
    ("help", "Hello! How can I help you?", 10),
    ("can you help me?", "Hello! How can I help you?", 10),
    ("i need help", "Can you tell me more?", 9),
    # Small talk
    ("how are you?", "How are you feeling today?", 10),
    ("what's up?", "That's interesting!", 8),
    ("tell me something", "I'm here to listen.", 9),
    ("what do you do?", "I'm here to listen.", 8),
    # Thanks
    ("thank you", "You're welcome!", 10),
    ("thanks", "You're welcome!", 10),
    ("thank you", "Goodbye!", 5),
    # Unknown/edge
    ("asdfgh", "I'm not sure I understand.", 10),
    ("", "I'm not sure I understand.", 10),
    ("random text", "Let's talk about something else.", 8),
    # Negative matches
    ("how are you?", "Goodbye!", 1),
    ("bye", "How are you feeling today?", 1),
    ("help", "Goodbye!", 2),
]

def softmax(xs):
    max_x = max(xs)  # for numerical stability
    exps = [math.exp(x - max_x) for x in xs]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def find_closest_input(user_input, known_inputs, cutoff=0.6):
    matches = difflib.get_close_matches(user_input, known_inputs, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def generate_response(user_input):
    # Simple keyword-based templates
    if "hello" in user_input.lower() or "hi" in user_input.lower():
        return random.choice([
            "Hi there! How can I help you?",
            "Hello! What brings you here today?",
            "Hey! How are you doing?"
        ])
    if "how are you" in user_input.lower():
        return random.choice([
            "I'm just a program, but I'm here to help!",
            "I'm doing well, thank you! How about you?",
            "I'm always ready to chat. How are you?"
        ])
    if "bye" in user_input.lower() or "goodbye" in user_input.lower():
        return random.choice([
            "Goodbye! Have a great day!",
            "See you later!",
            "Take care!"
        ])
    # Generic fallback templates
    templates = [
        "That's interesting, tell me more.",
        "Why do you say '{}'?".format(user_input),
        "How does that make you feel?",
        "Can you elaborate on '{}'?".format(user_input),
        "What else would you like to discuss?"
    ]
    return random.choice(templates)

def choose_response(user_input):
    # Try to find the closest known input
    known_inputs = list(ratings.keys())
    closest = find_closest_input(user_input, known_inputs)
    if closest:
        output_scores = ratings[closest]
    else:
        # If no close match, initialize as new
        for resp in possible_outputs:
            ratings[user_input][resp] = [0.5, 1]
        output_scores = ratings[user_input]
    raw_avgs = [output_scores[o][0] / output_scores[o][1] for o in possible_outputs]
    # Scale from [1, 10] to [0.01, 0.99]
    avgs = [0.01 + 0.98 * (min(max(avg, 1), 10) - 1) / 9 for avg in raw_avgs]
    probs = softmax(avgs)
    # 20% chance to generate a new response, 80% to use learned responses
    if random.random() < 0.2:
        return generate_response(user_input), True
    else:
        response = random.choices(possible_outputs, weights=probs, k=1)[0]
        return response, False

# Automated training (run more epochs for better accuracy)
for _ in range(15):  # Increased from 5 to 15
    for user_input, response, rating in training_data:
        if user_input not in ratings:
            for resp in possible_outputs:
                ratings[user_input][resp] = [0.5, 1]
        total, count = ratings[user_input][response]
        ratings[user_input][response] = [total + rating, count + 1]

def save_ratings(filename="ratings.json"):
    # Convert defaultdicts to normal dicts for JSON serialization
    serializable = {k: dict(v) for k, v in ratings.items()}
    with open(filename, "w") as f:
        json.dump(serializable, f)

def load_ratings(filename="ratings.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        for user_input, responses in data.items():
            for resp, vals in responses.items():
                ratings[user_input][resp] = vals

load_ratings()

print("Type 'q' or 'quit' to exit.")
while True:
    user_input = input("Input: ").strip()
    if user_input.lower() in ("q", "quit"):
        save_ratings()
        print("Exiting.")
        break
    response, is_generated = choose_response(user_input)
    if is_generated:
        print("Model (computer generated):", response)
    else:
        print("Model:", response)
    while True:
        try:
            rating = int(input("Rate this response (1-10): "))
            if 1 <= rating <= 10:
                break
            else:
                print("Please enter a number from 1 to 10.")
        except ValueError:
            print("Please enter a valid integer.")
    # Update ratings
    total, count = ratings[user_input][response]
    ratings[user_input][response] = [total + rating, count + 1] 