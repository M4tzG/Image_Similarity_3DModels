import json
import os
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm # For progress bars
from transformers import AutoImageProcessor, AutoModel
from datasets import Dataset, Image as HFImage # HFImage for Hugging Face Image type
from torchvision import transforms

# --- 1. Configuration and Model Loading (as in your script) ---
model_ckpt = "nateraw/vit-base-beans" # Or your preferred ViT model
processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval() # Set model to evaluation mode

# Batch size for processing embeddings
# Adjust based on your GPU memory / CPU capability
EMBEDDING_BATCH_SIZE = 32

# --- 2. Define Image Transformation Chain ---
# This chain preprocesses images for the ViT model.
# It should match the preprocessing used when the model was trained.
try:
    # Attempt to get size from processor, common for newer processors
    if hasattr(processor, 'size') and "shortest_edge" in processor.size:
        image_size = processor.size["shortest_edge"]
    elif hasattr(processor, 'size') and "height" in processor.size: # Fallback for older structures or specific models
        image_size = processor.size["height"]
    else: # Default if size info isn't directly available or in expected format
        image_size = 224 # A common default for ViT
except AttributeError:
    image_size = 224


transformation_chain = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ]
)

# --- 3. Load "Imagens modelo" based on your JSON map ---
# This corresponds to "json: 'modelo' : 'imagens'" feeding into the process
def load_dataset_from_json_map(json_file_path):
    """
    Loads image paths and their associated 3D model IDs from your JSON map.
    Returns a Hugging Face Dataset and a map to original 3D model paths.
    """
    image_file_paths = []
    associated_3d_model_ids = []
    source_3d_model_details = {} # Stores { '3d_model_id': 'path/to/3d_model.glb' }

    print(f"Loading image data from JSON map: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data_map = json.load(f)

    for model_id_3d, info in data_map.items():
        # Store the path to the original 3D model file for the "Match" step
        source_3d_model_details[model_id_3d] = info.get("model_3d_path", model_id_3d)

        for img_path in info["images"]:
            if not os.path.exists(img_path):
                print(f"Warning: Image path not found and skipped: {img_path}")
                continue
            image_file_paths.append(img_path)
            associated_3d_model_ids.append(model_id_3d)

    if not image_file_paths:
        raise ValueError("No valid image paths found. Check your JSON map and image file locations.")

    # Create a Hugging Face Dataset.
    # The "image" column will be loaded as PIL Images by HFImage().
    # "image_path" is kept for reference.
    # "id_3d_model" links the 2D image to its source 3D model.
    dataset_dict = {
        "image_path": image_file_paths, # For reference
        "image": image_file_paths,      # This will be loaded by HFImage()
        "id_3d_model": associated_3d_model_ids
    }
    hf_dataset = Dataset.from_dict(dataset_dict)
    # Cast the 'image' column to the Image feature to load images automatically
    hf_dataset = hf_dataset.cast_column("image", HFImage(decode=True))

    print(f"Loaded {len(hf_dataset)} images associated with {len(source_3d_model_details)} 3D models.")
    return hf_dataset, source_3d_model_details

# --- 4. Embedding Extraction Function (adapting your 'extract_embeddings') ---
# This creates "Embeddings (imagens modelo)"
def extract_embeddings_from_dataset(pytorch_model: torch.nn.Module):
    """Computes embeddings for images in the dataset."""
    device_of_model = pytorch_model.device

    def preprocess_batch(batch):
        # 'images' here are PIL Image objects because of HFImage(decode=True)
        pil_images = batch["image"]

        # Apply the transformation chain to each image
        # Ensure images are RGB, transformation_chain might expect this
        try:
            image_tensors_transformed = torch.stack(
                [transformation_chain(img.convert("RGB")) for img in pil_images]
            )
        except Exception as e:
            print(f"Error transforming images: {e}. Ensure images are valid PIL objects.")
            # Handle corrupt images: return empty or skip. Here, we'll raise to stop.
            # For robustness, you might want to filter out problematic images earlier.
            raise

        new_batch = {"pixel_values": image_tensors_transformed.to(device_of_model)}
        with torch.no_grad():
            embeddings = pytorch_model(**new_batch).last_hidden_state[:, 0].cpu() # CLS token embedding
        return {"embeddings": embeddings}

    return preprocess_batch

# --- 5. Cosine Similarity Function (as in your script) ---
def compute_cosine_scores(emb_one, emb_two):
    """Computes cosine similarity between two batches of embeddings."""
    # Ensure emb_one is (N, D) and emb_two is (1, D) or (M,D)
    # Cosine similarity expects inputs of same dimensions for pairwise, or one vs many
    if emb_one.ndim == 1: emb_one = emb_one.unsqueeze(0)
    if emb_two.ndim == 1: emb_two = emb_two.unsqueeze(0)

    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two, dim=1)
    return scores.numpy().tolist()


# --- 6. Fetch Similar Results Function (adapted 'fetch_similar') ---
# This performs "Cálculo de Similaridade."
def find_top_similar_images(
    query_pil_image: Image.Image,
    dataset_embeddings_tensor: torch.Tensor,
    dataset_info_list: list, # List of dicts: [{'image_path': str, 'id_3d_model': str}, ...]
    pytorch_model: torch.nn.Module,
    top_k: int = 5
):
    """
    Finds the top_k most similar images from the dataset to the query_pil_image.
    dataset_info_list should correspond row-wise to dataset_embeddings_tensor.
    """
    # Prepare the input query image for embedding computation ("Embeddings (input usuário)")
    query_transformed = transformation_chain(query_pil_image.convert("RGB")).unsqueeze(0)
    new_batch = {"pixel_values": query_transformed.to(pytorch_model.device)}

    with torch.no_grad():
        query_embedding = pytorch_model(**new_batch).last_hidden_state[:, 0].cpu()

    # Compute similarity scores with all candidate images
    similarity_scores = compute_cosine_scores(dataset_embeddings_tensor, query_embedding)

    # Combine scores with candidate image information
    results = []
    for i, score in enumerate(similarity_scores):
        results.append({
            "similarity_score": score, # "maior %" (for the top one)
            "dataset_image_path": dataset_info_list[i]["image_path"], # "caminho/imagem"
            "associated_3d_model_id": dataset_info_list[i]["id_3d_model"]
        })

    # Sort by similarity score in descending order
    results_sorted = sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    return results_sorted[:top_k]

# --- 7. Main Execution Logic ---
if __name__ == "__main__":
    # === Define Paths ===
    # !!! REPLACE WITH YOUR ACTUAL PATHS !!!
    your_json_map_file = "model_to_images_map.json" # Path to your JSON map
    your_input_image_path = "teste.jpg" # Path to the user's input image

    # === Step A: Load "Imagens modelo" and their 3D model associations ===
    # This also populates `source_3d_model_paths_map` for the final "Match" step.
    try:
        model_images_dataset, source_3d_model_paths_map = load_dataset_from_json_map(your_json_map_file)
    except FileNotFoundError:
        print(f"Error: JSON map file not found at {your_json_map_file}. Please check the path.")
        exit()
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        exit()

    # === Step B: Compute "Embeddings (imagens modelo)" ===
    print("Extracting embeddings for the dataset images...")
    embedding_extractor_fn = extract_embeddings_from_dataset(model)
    # The .map() function will apply `embedding_extractor_fn` to batches of images.
    # It automatically handles the 'image' column (PIL Images) and adds an 'embeddings' column.
    model_images_dataset_with_embeddings = model_images_dataset.map(
        embedding_extractor_fn, batched=True, batch_size=EMBEDDING_BATCH_SIZE
    )
    print("Embeddings extracted.")

    # Prepare embeddings tensor and corresponding info list for quick lookup
    all_dataset_embeddings = torch.tensor(np.array(model_images_dataset_with_embeddings["embeddings"]))

    # Create a list of dictionaries holding the path and 3D model ID for each embedding
    # This ensures the order matches all_dataset_embeddings
    dataset_image_info = [
        {"image_path": item["image_path"], "id_3d_model": item["id_3d_model"]}
        for item in model_images_dataset_with_embeddings
    ]

    # === Step C: Load "Input Image" and find similar models ===
    try:
        input_query_image_pil = Image.open(your_input_image_path)
    except FileNotFoundError:
        print(f"Error: Input image not found at {your_input_image_path}. Please check the path.")
        exit()

    print(f"\nFinding models similar to: {your_input_image_path}")
    top_k_results = find_top_similar_images(
        query_pil_image=input_query_image_pil,
        dataset_embeddings_tensor=all_dataset_embeddings,
        dataset_info_list=dataset_image_info,
        pytorch_model=model,
        top_k=3 # Get top 3 matches
    )

    # === Step D: "Match." - Display results and the corresponding 3D model ===
    print("\n--- Top Matches ---")
    if not top_k_results:
        print("No similar images found.")
    else:
        for i, result in enumerate(top_k_results):
            similarity_percentage = result['similarity_score'] * 100
            matched_2d_image = result['dataset_image_path']
            associated_3d_id = result['associated_3d_model_id']

            # Retrieve the original 3D model path/identifier using the map
            original_3d_model_ref = source_3d_model_paths_map.get(associated_3d_id, "N/A (ID not found in map)")

            print(f"\nRank {i+1}:")
            print(f"  Similarity: {similarity_percentage:.2f}%")
            print(f"  Most Similar 2D Image (from dataset): {matched_2d_image}")
            print(f"  Associated 3D Model ID: {associated_3d_id}")
            print(f"  --> Original 3D Model Path/Reference: {original_3d_model_ref}")

    print("\nDone.")
