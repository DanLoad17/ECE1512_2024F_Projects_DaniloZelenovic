import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel, AutoTokenizer, AutoModelForCausalLM
import time
from fvcore.nn import FlopCountAnalysis

class LLaVA(nn.Module):
    def __init__(self, vision_encoder, language_model, output_dim=768, token_slice_ratio=0.5):
        super(LLaVA, self).__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.output_layer = nn.Linear(768 + 768, output_dim)  # Fix to match the combined feature dimension
        self.token_slice_ratio = token_slice_ratio 

    def forward(self, image, text):
        image_features = self.vision_encoder(image).last_hidden_state 
        image_features = image_features.mean(dim=1)  

        num_tokens = text.size(1)
        slice_size = int(num_tokens * self.token_slice_ratio) 
        sliced_text = text[:, :slice_size] 
        output = self.language_model(sliced_text, output_hidden_states=True)
        text_embeddings = output.hidden_states[-1] 
        text_embeddings = text_embeddings[:, 0, :] 
        combined_features = torch.cat((image_features, text_embeddings), dim=1)  # (batch_size, 768 + 768)
        output = self.output_layer(combined_features) 
        return output

def load_models():
    vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vision_model.eval()

    language_model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True)
    language_model.eval() 

    return vision_model, language_model

def prepare_input_data():
    image = torch.randn(1, 3, 224, 224)  # Simulate a batch of images
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = tokenizer("This is a test sentence.", return_tensors="pt")["input_ids"]
    
    return image, text

def compute_mse_loss(output, target):
    loss_fct = torch.nn.MSELoss()
    loss = loss_fct(output, target)
    return loss

def measure_efficiency(model, image, text):
    start_time = time.time()
    output = model(image, text)
    latency = time.time() - start_time
    flops = FlopCountAnalysis(model, (image, text)).total()
    return output, latency, flops

def baseline_experiment():
    vision_model, language_model = load_models()

    baseline_llava = LLaVA(vision_model, language_model)
    image_sample, text_sample = prepare_input_data()

    print("Running Baseline Experiment...")
    output, latency, flops = measure_efficiency(baseline_llava, image_sample, text_sample)
    print(f"Baseline Latency: {latency:.4f} seconds")
    print(f"Baseline FLOPs: {flops}")


    improved_llava = LLaVA(vision_model, language_model, token_slice_ratio=0.5) 

    print("Running Improved Efficiency Experiment with Token Slicing...")
    improved_output, improved_latency, improved_flops = measure_efficiency(improved_llava, image_sample, text_sample)
    print(f"Improved Efficiency Latency: {improved_latency:.4f} seconds")
    print(f"Improved Efficiency FLOPs: {improved_flops}")

    target_tensor = torch.randn_like(output)
    baseline_loss = compute_mse_loss(output, target_tensor)
    print(f"Baseline MSE Loss: {baseline_loss.item():.4f}")
    improved_loss = compute_mse_loss(improved_output, target_tensor) 
    print(f"Improved Efficiency MSE Loss: {improved_loss.item():.4f}")

if __name__ == "__main__":
    baseline_experiment()
