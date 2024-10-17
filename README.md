# Flood Response Chatbot

This project implements a conversational AI chatbot fine-tuned on flood-related information using the DialoGPT-medium model. The chatbot can provide guidance and answer user queries about flood safety, preparation, and general information. It consists of two main scripts:
- `Training.py`: For training the chatbot model on custom dialog data.
- `Prediction.py`: For generating responses based on user inputs using the trained model.

## Project Structure

```
├── Training.py          # Script to train the chatbot
├── Prediction.py        # Script to generate chatbot responses
├── conversations.json   # Sample training data in JSON format
├── user_input.json      # Sample user input data in JSON format
├── flood-dialoGPT/      # Directory where the trained model and tokenizer are saved
├── logs/                # Directory for storing training logs
└── README.md            # Project documentation
```

## Requirements

Before you start, make sure you have the following dependencies installed:

- Python 3.8+
- `transformers`
- `torch`
- `json`

You can install the required Python libraries using the following command:
```bash
pip install torch transformers
```

## Training the Chatbot

To train the chatbot, use the `Training.py` script. The training data should be in a JSON file (`conversations.json`) with the following format:
```json
[
    {
        "input": "What should I do during a flood?",
        "response": "Stay indoors, avoid flooded areas, and listen to emergency services."
    },
    {
        "input": "How can I prepare for a flood?",
        "response": "Keep emergency supplies, know your evacuation routes, and stay informed."
    },
    {
        "input": "What are the common signs of a flood?",
        "response": "Rising water levels, continuous rain, and overflowing rivers."
    }
]
```

### Running the Training Script

```bash
python Training.py
```

This will train the model based on the provided data and save the trained model and tokenizer in the `./flood-dialoGPT` directory.

### Adjustable Hyperparameters in `Training.py`

You can adjust the following hyperparameters for training:
- `per_device_train_batch_size`: Batch size per device. Increase if using a larger dataset or stronger hardware. (e.g., 4, 8, 16)
- `num_train_epochs`: Number of epochs. Increase for more extensive training on larger datasets. (e.g., 3-5)
- `save_steps`: Frequency of checkpoint saving during training. Increase for longer training. (e.g., 100, 1000)
- `save_total_limit`: Maximum number of checkpoints to keep. (e.g., 2, 3)
- `logging_steps`: Frequency of logging during training.

```python
training_args = TrainingArguments(
    output_dir="./flood-dialoGPT",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=10,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10
)
```

## Generating Responses

To generate responses from the trained chatbot, use the `Prediction.py` script. The user inputs should be in `user_input.json` and follow this format:
```json
[
    "How can I stay safe during a flood?",
    "How to respond to a flood?",
    "Why do floods happen?"
]
```

### Running the Prediction Script

```bash
python Prediction.py
```

### Sample Output
```
User: How can I stay safe during a flood?
Bot: Stay indoors, avoid flooded areas, and listen to emergency services.

User: How to respond to a flood?
Bot: Ensure safety, call emergency services, and follow the authorities' guidelines.

User: Why do floods happen?
Bot: Floods occur due to heavy rainfall, river overflows, or natural disasters like hurricanes.
```

### Adjustable Hyperparameters in `Prediction.py`

You can adjust the following hyperparameters for response generation:
- `max_length`: Maximum length of the response. Increase if you need longer answers. (e.g., 150-200)
- `no_repeat_ngram_size`: Set to avoid repeating phrases.
- `top_k`: Adjust for more or less diversity. Lower for more focused answers. (e.g., 30, 50)
- `top_p`: Nucleus sampling. Adjust for more or less variation. (e.g., 0.9, 0.95)
- `temperature`: Controls randomness. Lower for more deterministic answers, higher for more varied. (e.g., 0.5, 1.0)

```python
output = model.generate(
    input_ids,
    max_length=100,
    pad_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)
```

## Notes
- Ensure that the `./flood-dialoGPT` and `./logs` directories exist or are created by the script.
- Make sure your JSON files (`conversations.json` and `user_input.json`) are correctly formatted to avoid any errors during training or prediction.
- Adjust the hyperparameters as needed based on the dataset size and computational resources available.

## License
This project is open source and available under the MIT License. Feel free to use, modify, and share it.
