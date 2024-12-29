import json
import re
import time
from functools import partial
from pathlib import Path

import tiktoken
import torch
import argparse

from tqdm import tqdm

from libs.training import gpt_load_params, train_model_simple
from libs.download_load_data import download_load_files
from libs.dataloader import create_dataloader
from libs.datasets import format_input
from libs.model_evaluate import calc_loss_loader, plot_values
from libs.utils import generate, text_to_token_ids, token_ids_to_text

#if __name__ == "__main":



parser = argparse.ArgumentParser(
    description="Finetune a GPT model for instructions"
)
parser.add_argument(
    "--test_mode",
    default=False,
    action="store_true",
    help=("This flag runs the model in test mode for internal testing purposes"
          ".")
)
args = parser.parse_args()
#######################################
# Download and prepare dataset
#######################################
file_path = Path("../../datasets/instruction")
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
data = download_load_files(file_path / "instruction-data.json", url)

#print(data)

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)  # 10% for testing

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# Use very small subset for testing purposes
if args.test_mode:
    train_data = train_data[:590]
    val_data = val_data[:300]
    test_data = test_data[:300]
    mode = "test_mode"
    model_name = "Simple Test"
else:
    mode = "train"
    model_name = "gpt2-small (124M)"

print(test_data)


with open(file_path / "train.json", "w") as js_file:
    json.dump(train_data, js_file, indent=4)
with open(file_path / "validation.json", "w") as js_file:
    json.dump(val_data, js_file, indent=4)

with open(file_path / "test.json", "w") as js_file:
    json.dump(test_data, js_file, indent=4)


print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))
print(50 * "-")

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(50 * "-")

train_loader = create_dataloader(file_path / "train.json", batch_size=4,
                                 max_length=1024,
                                 stride=128,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=0,
                                 dataset_type="instruction",
                                 device=device)

val_loader = create_dataloader(file_path / "validation.json", batch_size=4,
                               max_length=1024,
                               stride=128,
                               shuffle=True,
                               drop_last=True,
                               num_workers=0,
                               dataset_type="instruction",
                               device=device)

#######################################
# Load pretrained model
#######################################

model, BASE_CONFIG = gpt_load_params(model_name, device, mode)
model.to(device)

#######################################
# Finetuning the model
#######################################

print("Initial losses")
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("   Training loss:", train_loss)
print("   Validation loss:", val_loss)

start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 2

torch.manual_seed(123)
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_values(epochs_tensor, tokens_seen, train_losses, val_losses, figsize=(12, 6))
print(50 * "-")

#######################################
# Saving results
#######################################
print("Generating responses")
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

test_data_path = "instruction-data-with-response-standalone.json"
with open(test_data_path, "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
print(f"Responses saved as {test_data_path}")

file_name = f"{re.sub(r'[ ()]', '', model_name)}-sft-standalone.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")