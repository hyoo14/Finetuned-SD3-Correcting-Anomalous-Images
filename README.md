# Finetuned-SD3-Correcting-Anomalous-Images

## Introduction

The Finetuned-SD3-Correcting-Anomalous-Images project aims to address the issue of anomalous image generation in text-to-image models. By fine-tuning the Stable Diffusion 3 model using the DreamBooth technique, this project seeks to improve the quality of images generated from specific prompts like "lying on the grass/street". The fine-tuned model shows enhanced performance in visual assessments and key metrics such as SSIM, PSNR, and FID, while also receiving positive user feedback. This work aims to increase the practicality and reliability of text-to-image models in various applications.



### Model

You can find the trained model on Hugging Face at the following link:

[Trained SD3 Model](https://huggingface.co/hyoo14/trained-sd3-lora-miniature)


### Colab Notebook

You can find the Colab notebook at the following link:

[Colab Notebook](https://colab.research.google.com/drive/1RnGaE_b3k6V9e9-kO4eO2sLr9Yg2g6ZL?usp=sharing)



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





<br/>
<br/>

### Foundation Model:     

This project utilized the [stabilityai](https://huggingface.co/stabilityai/) implementation of **stable-diffusion-3-medium-diffusers** for this project.  


### Foundation Code:      

This project utilized the [Dreambooth](https://dreambooth.github.io/) implementation.  
