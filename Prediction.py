import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./flood-dialoGPT"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load user input from a JSON file
user_input_file = "user_input.json"
with open(user_input_file, "r") as file:
    user_input_data = json.load(file)
user_input = user_input_data["user_input"]

# Function to generate a response (Adjustable Hyperparameters)
def generate_response(prompt):  # Controls response length; increase for longer outputs
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    
    # Generate a response using the model
    output = model.generate(
        input_ids,
        max_length=100,          # Maximum length of response
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # Avoid repeating phrases
        do_sample=True,          # Enable sampling to get more diverse outputs
        top_k=50,                # Limit to top-k sampling; can be adjusted for variation
        top_p=0.95,              # Nucleus sampling; can be adjusted for variation
        temperature=0.7          # Adjust randomness; lower for more deterministic outputs
        # Suggested Values:
        # - `max_length`: Increase to 150-200 if you need longer responses.
        # - `top_k` and `top_p`: Adjust for more or less diversity. Lower `top_k` or `top_p` for more focused answers.
        # - `temperature`: Lower to 0.5 for more deterministic and less creative responses; increase to 1.0 for more varied responses.
    )
    
    # Decode the generated tokens
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Example usage
response = generate_response(user_input)
print(f"User: {user_input}")
print(f"Bot: {response}")
