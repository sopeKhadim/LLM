import time

import pandas as pd
import tiktoken
import torch
from pathlib import Path
from libs.download_load_data import download_and_unzip_spam_data
from libs.utils import create_balanced_dataset
from libs.utils import random_split
from libs.dataloader import create_dataloader
from libs.training import gpt_load_params, train_classifier_simple
from libs.model_evaluate import plot_values

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune a GPT model for spam_classifier"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    args = parser.parse_args()

    ########################################
    # Download and prepare dataset
    ########################################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "../../datasets/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=args.test_mode)
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)

    cleaned_data_path = Path("../../datasets/spam")

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv(cleaned_data_path / "train.csv", index=None)
    validation_df.to_csv(cleaned_data_path / "validation.csv", index=None)
    test_df.to_csv(cleaned_data_path / "test.csv", index=None)

    ########################################
    # Create data loaders
    ########################################

    if args.test_mode:
        mode = "test_mode"
        model_name = "Simple Test"
        device = "cpu"
    else:
        mode="train"
        model_name = "gpt2-small (124M)"
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Hardware Device : {device}")
    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader = create_dataloader(cleaned_data_path / "train.csv", batch_size=4,
                      max_length=None,
                      stride=128,
                      shuffle=True,
                      drop_last=True,
                      num_workers=0,
                      dataset_type="spam",
                      device=device)

    val_loader = create_dataloader(cleaned_data_path / "validation.csv", batch_size=4,
                      max_length=None,
                      stride=128,
                      shuffle=True,
                      drop_last=True,
                      num_workers=0,
                      dataset_type="spam",
                      device=device)

    test_dataset = create_dataloader(cleaned_data_path / "test.csv", batch_size=4,
                                   max_length=None,
                                   stride=128,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=0,
                                   dataset_type="spam",
                                   device=device)


    model,  BASE_CONFIG = gpt_load_params(model_name, device, mode)

    ########################################
    # Modify and pretrained model
    ########################################

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    model.to(device)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

        ########################################
        # Finetune modified model
        ########################################

        start_time = time.time()
        torch.manual_seed(123)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

        num_epochs = 5
        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=50, eval_iter=5, type_training="classification"
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        ########################################
        # Plot results
        ########################################

        # loss plot
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
        plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

        # accuracy plot
        epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
        plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

        # Save model
        torch.save(model.state_dict(), "models/spam_classifier/review_classifier.pth")


