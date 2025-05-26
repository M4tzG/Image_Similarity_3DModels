import json
import os
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel
from datasets import Dataset, Image as HFImage
from torchvision import transforms


# --- 1. configuracao geral ---
model_ckpt = "nateraw/vit-base-beans" # ajustar posteriormente (esse foi otimizado para o dataset beans)
processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# processamento dos embeddings
EMBEDDING_BATCH_SIZE = 32


# --- 2. Image Transformation Chain ---
try:
    if hasattr(processor, 'size') and "shortest_edge" in processor.size:
        image_size = processor.size["shortest_edge"]
    elif hasattr(processor, 'size') and "height" in processor.size:
        image_size = processor.size["height"]
    else: 
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


# --- 3. load das imagens utilizadas para o modelo  ---
# corresponde ao "json: 'modelo' : 'imagens'" 
def load_dataset_from_json_map(json_file_path):
    """
    Load do caminho para as imagens, o qual foi definido no 'renders_map.json'
    retorna um 'Hugging Face Dataset'
    """
    image_file_paths = []
    associated_3d_model_ids = []
    source_3d_model_details = {} # { '3d_model_id': 'path/to/3d_model.glb' }

    print(f"Carregando imagem de: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data_map = json.load(f)

    for model_id_3d, info in data_map.items():
        source_3d_model_details[model_id_3d] = info.get("model_3d_path", model_id_3d)

        for img_path in info["images"]:
            if not os.path.exists(img_path):
                print(f"Warning: imagem nao encontrada: {img_path}")
                continue
            image_file_paths.append(img_path)
            associated_3d_model_ids.append(model_id_3d)

    if not image_file_paths:
        raise ValueError("Caminhos para imagens nao encontrados.")

    # Hugging Face Dataset.
    dataset_dict = {
        "image_path": image_file_paths, # ref
        "image": image_file_paths,      # by HFImage()
        "id_3d_model": associated_3d_model_ids
    }
    hf_dataset = Dataset.from_dict(dataset_dict)
    hf_dataset = hf_dataset.cast_column("image", HFImage(decode=True))

    print(f"Loaded {len(hf_dataset)} images associated with {len(source_3d_model_details)} 3D models.")
    return hf_dataset, source_3d_model_details


# --- 4. Embedding Extraction ---
def extract_embeddings_from_dataset(pytorch_model: torch.nn.Module):
    """
    Computa os embeddings das imagens do dataset
    """
    device_of_model = pytorch_model.device

    def preprocess_batch(batch):
        pil_images = batch["image"]
        try:
            image_tensors_transformed = torch.stack(
                [transformation_chain(img.convert("RGB")) for img in pil_images]
            )
        except Exception as e:
            print(f"Error transforming images: {e}. Ensure images are valid PIL objects.")
            raise

        new_batch = {"pixel_values": image_tensors_transformed.to(device_of_model)}
        with torch.no_grad():
            embeddings = pytorch_model(**new_batch).last_hidden_state[:, 0].cpu() # CLS token embedding
        return {"embeddings": embeddings}

    return preprocess_batch


# --- 5. Similaridade de cosseno ---
def compute_cosine_scores(emb_one, emb_two):
    """
    Calcula a similaridade de cosseno entre as imagens
    """
    if emb_one.ndim == 1: emb_one = emb_one.unsqueeze(0)
    if emb_two.ndim == 1: emb_two = emb_two.unsqueeze(0)

    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two, dim=1)
    return scores.numpy().tolist()


# --- 6. Resultados ---
def find_top_similar_images(
    query_pil_image: Image.Image,
    dataset_embeddings_tensor: torch.Tensor,
    dataset_info_list: list, # [{'image_path': str, 'id_3d_model': str}, ...]
    pytorch_model: torch.nn.Module,
    top_k: int = 5
):
    """
    Retorna as top imagens mais similares ao input
    """
    query_transformed = transformation_chain(query_pil_image.convert("RGB")).unsqueeze(0)
    new_batch = {"pixel_values": query_transformed.to(pytorch_model.device)}

    with torch.no_grad():
        query_embedding = pytorch_model(**new_batch).last_hidden_state[:, 0].cpu()

    similarity_scores = compute_cosine_scores(dataset_embeddings_tensor, query_embedding)

    results = []
    for i, score in enumerate(similarity_scores):
        results.append({
            "similarity_score": score, # "maior %" (for the top one)
            "dataset_image_path": dataset_info_list[i]["image_path"], # "caminho/imagem"
            "associated_3d_model_id": dataset_info_list[i]["id_3d_model"]
        })

    # sort
    results_sorted = sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    return results_sorted[:top_k]

# --- 7. Main ---
if __name__ == "__main__":
    # === Paths ===
    your_json_map_file = "renders_map.json" # caminho para o json
    your_input_image_path = "Test_Inputs/desk.jpg" # input image

    # ================
    # Load Imgs
    try:
        model_images_dataset, source_3d_model_paths_map = load_dataset_from_json_map(your_json_map_file)
    except FileNotFoundError:
        print(f"Error: JSON map file not found at {your_json_map_file}. Please check the path.")
        exit()
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        exit()

    # ================
    # Embdeddings
    print("Extracting embeddings for the dataset images...")
    embedding_extractor_fn = extract_embeddings_from_dataset(model)
    model_images_dataset_with_embeddings = model_images_dataset.map(
        embedding_extractor_fn, batched=True, batch_size=EMBEDDING_BATCH_SIZE
    )
    print("Embeddings extracted.")

    all_dataset_embeddings = torch.tensor(np.array(model_images_dataset_with_embeddings["embeddings"]))

    dataset_image_info = [
        {"image_path": item["image_path"], "id_3d_model": item["id_3d_model"]}
        for item in model_images_dataset_with_embeddings
    ]

    # ================
    # Load input img e calcula similares
    try:
        input_query_image_pil = Image.open(your_input_image_path)
    except FileNotFoundError:
        print(f"Error: Input image not found at {your_input_image_path}. Please check the path.")
        exit()

    print(f"\nProcurando similares: {your_input_image_path}")
    top_k_results = find_top_similar_images(
        query_pil_image=input_query_image_pil,
        dataset_embeddings_tensor=all_dataset_embeddings,
        dataset_info_list=dataset_image_info,
        pytorch_model=model,
        top_k=3 
    )

    # ================
    # print top
    print("\n--- Top Matches ---")
    if not top_k_results:
        print("Sem resultados.")
    else:
        for i, result in enumerate(top_k_results):
            similarity_percentage = result['similarity_score'] * 100
            matched_2d_image = result['dataset_image_path']
            associated_3d_id = result['associated_3d_model_id']

            original_3d_model_ref = source_3d_model_paths_map.get(associated_3d_id, "N/A (ID not found in map)")

            print(f"\nRank {i+1}:")
            print(f"  Similaridade: {similarity_percentage:.2f}%")
            print(f"  Mais similar (dataset): {matched_2d_image}")
            print(f"  3D Model ID: {associated_3d_id}")
            print(f"  --> Original 3D Model Path/Reference: {original_3d_model_ref}")

