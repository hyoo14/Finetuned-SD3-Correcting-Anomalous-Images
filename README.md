# Finetuned-SD3-Correcting-Anomalous-Images

### Model

You can find the trained model on Hugging Face at the following link:

[Trained SD3 Model](https://huggingface.co/hyepark/trained-sd3)


### Colab Notebook

You can find the Colab notebook at the following link:

[Colab Notebook](https://colab.research.google.com/drive/1asdfb3k6V9e9-asdfasdfL?usp=sharing)



## Usage

### Training

1. Clone the repository:

    ```sh
    git clone https://github.com/hyoo14/Finetuned-SD3-Correcting-Anomalous-Images.git
    ```

2. Navigate to the project directory:

    ```sh
    cd Finetuned-SD3-Correcting-Anomalous-Images
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Login to Hugging Face CLI:

    ```sh
    huggingface-cli login --token <user_token>
    ```

5. Navigate to the training directory:

    ```sh
    cd training
    ```

6. Compute the embeddings:

    ```sh
    python compute_embeddings.py
    ```

7. Login to `wandb`:

    ```sh
    WANDB_API_KEY=<user_key> wandb login
    ```

8. Start the training process:

    ```sh
    accelerate launch train.py \
      --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers"  \
      --instance_data_dir="../data" \
      --data_df_path="sample_embeddings.parquet" \
      --output_dir="trained-sd3-lora-miniature" \
      --mixed_precision="fp16" \
      --instance_prompt="lying on the grass/street" \
      --resolution=1024 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=4 --gradient_checkpointing \
      --use_8bit_adam \
      --learning_rate=1e-4 \
      --report_to="wandb" \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --max_train_steps=500 \
      --seed="0"
    ```

### Inference

To run the inference, use the following command:

```sh
python ../inference/inference.py
```
