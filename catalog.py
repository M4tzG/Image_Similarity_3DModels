import os
import json

def cat_renders(main_folder="Renders", json_output="Image_Similarity/renders_map.json"):
    """
    'Cataloga' imagens de renderizacao de subpastas em um arquivo JSON.

    Args:
        main_folder (str): O nome da pasta contendo as subpastas dos objetos renderizados
        json_output (str): arquivo json gerado.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Pega o diretorio do script
    render_path = os.path.join(script_dir, main_folder)

    data = {}
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']

    if not os.path.isdir(render_path):
        print(f"ERRO: A pasta '{main_folder}' nao foi encontrada em '{script_dir}'.")
        return

    # pastas em 'Render'
    try:
        obj_name = [nome for nome in os.listdir(render_path)
                         if os.path.isdir(os.path.join(render_path, nome))]
    except OSError as e:
        print(f"ERRO ao acessar a pasta '{render_path}': {e}")
        return
        
    if not obj_name:
        print(f"Nenhuma subpasta (objeto) encontrada em '{render_path}'.")
        return

    print(f"Encontradas as seguintes pastas de objetos: {', '.join(obj_name)}")

    for name in obj_name:
        id_render = name  # nome da pasta eh o id
        obj_dir_path = os.path.join(render_path, name)
        
        img_list = []
        try:
            imgs_obj_dir = os.listdir(obj_dir_path)
        except OSError as e:
            print(f"AVISO: impossivel acessar os arquivos na pasta '{obj_dir_path}': {e}")
            continue

        for img_name in imgs_obj_dir:
            if os.path.isfile(os.path.join(obj_dir_path, img_name)):
                base_name, extension = os.path.splitext(img_name)
                if extension.lower() in valid_extensions:
                    abs_path = os.path.join(render_path, name, img_name)
                    img_list.append(abs_path.replace("\\", "/"))
        
        if not img_list:
            print(f"AVISO: Nenhuma imagem encontrada na pasta '{obj_dir_path}'.")
        
        data[id_render] = {
            "model_3d_path": "caminho/para/o/modelo/3d",
            "images": sorted(img_list)
        }
        print(f"Processada pasta: '{name}', {len(img_list)} imagens encontradas.")

    # dict to json
    json_output = os.path.join(script_dir, json_output)
    try:
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nArquivo JSON '{json_output}' criado com sucesso em '{script_dir}'.")
        print(f"Total de {len(data)} IDs de render processados.")
    except IOError as e:
        print(f"ERRO ao salvar o arquivo JSON '{json_output}': {e}")
    except TypeError as e:
        print(f"ERRO de tipo ao tentar serializar para JSON: {e}")


if __name__ == "__main__":
    # cat_renders(main_folder="minhas_fotos", json_output="catalogo.json")
    cat_renders()
