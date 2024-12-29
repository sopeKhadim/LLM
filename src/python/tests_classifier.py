from pathlib import Path

finetuned_model_path = Path("models/spam_classifier/review_classifier.pth")
if not finetuned_model_path.exists():
    print(
        f"Could not find '{finetuned_model_path}'.\n"
        "Run the `finetune_classification.py` to finetune and save the finetuned model."
    )
from libs.gpt_model import GPTModel


base_config = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"

base_config.update(model_configs[model_name])

# Initialize base model
model = GPTModel(base_config)

import torch

# Convert model to classifier as in section 6.5 in ch06.ipynb
num_classes = 2
model.out_head = torch.nn.Linear(in_features=base_config["emb_dim"], out_features=num_classes)

# Then load pretrained weights
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.load_state_dict(torch.load(finetuned_model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# This function was implemented in ch06.ipynb
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor.to(device))[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(f"Text1 : {text_1}")
pred_text1 = classify_review(
    text_1, model, tokenizer, device, max_length=120
)
print(f" Prediction text_1 : {pred_text1}")

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(f"Text2 : {text_2}")

pred_text2 = classify_review(
    text_2, model, tokenizer, device, max_length=120
)
print(f"Prediction text_2: {pred_text2}")