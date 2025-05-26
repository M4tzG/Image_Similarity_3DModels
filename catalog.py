import os
import json

def catalogar_renders(pasta_principal_renders="renders", arquivo_json_saida="Image_Similarity/renders_map.json"):
    """
    Cataloga imagens de renderização de subpastas em um arquivo JSON.

    Args:
        pasta_principal_renders (str): O nome da pasta contendo as subpastas dos objetos renderizados.
                                       Espera-se que esta pasta esteja no mesmo diretório do script.
        arquivo_json_saida (str): O nome do arquivo JSON a ser gerado.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Pega o diretório do script
    caminho_pasta_renders = os.path.join(script_dir, pasta_principal_renders)

    dados_finais = {}
    extensoes_imagem_validas = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']

    if not os.path.isdir(caminho_pasta_renders):
        print(f"ERRO: A pasta '{pasta_principal_renders}' não foi encontrada em '{script_dir}'.")
        print("Por favor, crie a pasta 'renders' e coloque as subpastas dos seus objetos dentro dela,")
        print("ou ajuste o parâmetro 'pasta_principal_renders' no script.")
        return

    # Lista todas as entradas na pasta 'renders'
    try:
        nomes_objetos = [nome for nome in os.listdir(caminho_pasta_renders)
                         if os.path.isdir(os.path.join(caminho_pasta_renders, nome))]
    except OSError as e:
        print(f"ERRO ao acessar a pasta '{caminho_pasta_renders}': {e}")
        return
        
    if not nomes_objetos:
        print(f"Nenhuma subpasta (objeto) encontrada em '{caminho_pasta_renders}'.")
        return

    print(f"Encontradas as seguintes pastas de objetos: {', '.join(nomes_objetos)}")

    for nome_objeto in nomes_objetos:
        id_render = nome_objeto  # O nome da pasta é o ID
        caminho_pasta_objeto = os.path.join(caminho_pasta_renders, nome_objeto)
        
        lista_imagens = []
        try:
            arquivos_na_pasta_objeto = os.listdir(caminho_pasta_objeto)
        except OSError as e:
            print(f"AVISO: Não foi possível acessar os arquivos na pasta '{caminho_pasta_objeto}': {e}. Pulando esta pasta.")
            continue

        for nome_arquivo in arquivos_na_pasta_objeto:
            # Verifica se é um arquivo e se tem uma extensão de imagem válida
            if os.path.isfile(os.path.join(caminho_pasta_objeto, nome_arquivo)):
                nome_base, extensao = os.path.splitext(nome_arquivo)
                if extensao.lower() in extensoes_imagem_validas:
                    # Cria o caminho relativo a partir da pasta onde o script está
                    # ex: renders/NomeDoObjeto/imagem.png
                    caminho_relativo_imagem = os.path.join(pasta_principal_renders, nome_objeto, nome_arquivo)
                    # Normaliza as barras para o padrão do OS (opcional, mas bom para consistência)
                    lista_imagens.append(caminho_relativo_imagem.replace("\\", "/")) 
        
        if not lista_imagens:
            print(f"AVISO: Nenhuma imagem encontrada na pasta '{caminho_pasta_objeto}'.")
        
        dados_finais[id_render] = {
            "model_3d_path": "caminho/para/o/modelo/3d",  # Placeholder
            "images": sorted(lista_imagens) # Ordena para consistência
        }
        print(f"Processada pasta: '{nome_objeto}', {len(lista_imagens)} imagens encontradas.")

    # Salvar o dicionário como JSON
    caminho_arquivo_json = os.path.join(script_dir, arquivo_json_saida)
    try:
        with open(caminho_arquivo_json, 'w', encoding='utf-8') as f:
            json.dump(dados_finais, f, ensure_ascii=False, indent=2) # indent=2 para formatação legível
        print(f"\nArquivo JSON '{arquivo_json_saida}' criado com sucesso em '{script_dir}'.")
        print(f"Total de {len(dados_finais)} IDs de render processados.")
    except IOError as e:
        print(f"ERRO ao salvar o arquivo JSON '{caminho_arquivo_json}': {e}")
    except TypeError as e:
        print(f"ERRO de tipo ao tentar serializar para JSON: {e}")


if __name__ == "__main__":
    # Você pode alterar o nome da pasta principal de renders e o nome do arquivo de saída aqui, se desejar.
    # Exemplo: catalogar_renders(pasta_principal_renders="minhas_fotos", arquivo_json_saida="catalogo.json")
    catalogar_renders()
