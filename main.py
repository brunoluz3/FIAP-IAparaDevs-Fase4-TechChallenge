from dotenv import load_dotenv
from openai import OpenAI
import boto3
import os

import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
from deepface import DeepFace

from flask import Flask, request, jsonify
import json

from Functions.ConvertVideo import ConvertVideo
from Functions.AwsServices import AwsServices
from Functions.SaveFile import SaveFile
from Functions.Transcription import Transcription

app = Flask(__name__)

def CarregarBancoImagens():
    bancoImagens = f"Arquivos\\BancoImagens"

    rosto_encondig = []
    nome_rosto = []

    for filename in os.listdir(bancoImagens):
        # Verificar se o arquivo é uma imagem
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Carregar a imagem
            image_path = os.path.join(bancoImagens, filename)
            image = face_recognition.load_image_file(image_path)
            # Obter as codificações faciais (assumindo uma face por imagem)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                face_encoding = face_encodings[0]
                # Extrair o nome do arquivo, removendo o sufixo numérico e a extensão
                name = os.path.splitext(filename)[0][:-1]
                # Adicionar a codificação e o nome às listas
                rosto_encondig.append(face_encoding)
                nome_rosto.append(name)

    return rosto_encondig, nome_rosto

def DetectarFaceEmocoes(video_original, video_reconhecimento):
    rosto_encondig, nome_rosto = CarregarBancoImagens()

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_original)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(video_reconhecimento, fourcc, fps, (width, height))

    # Loop para processar cada frame do vídeo com barra de progresso
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Analisar o frame para detectar faces e expressões
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Obter as localizações e codificações das faces conhecidas no frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Inicializar uma lista de nomes para as faces detectadas
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(rosto_encondig, face_encoding)
            name = "Desconhecido"
            face_distances = face_recognition.face_distance(rosto_encondig, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = nome_rosto[best_match_index]
            face_names.append(name)

        # Iterar sobre cada face detectada pelo DeepFace
        for face in result:
            # Obter a caixa delimitadora da face
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
            # Obter a emoção dominante
            dominant_emotion = face['dominant_emotion']

            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Escrever a emoção dominante acima da face
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Associar a face detectada pelo DeepFace com as faces conhecidas
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if x <= left <= x + w and y <= top <= y + h:
                    # Escrever o nome abaixo da face
                    cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    break
        
        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def AnalisarInformacoesVideo(video, audio, arquivoTranscricao, arquivoSentimento):
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    comprehend = boto3.client("comprehend", os.environ.get("region"))    

    converterVideo = ConvertVideo()
    converterVideo.ConvertVideoToAudio(video, audio)    

    transcricaoAudio = Transcription()
    transcricao = transcricaoAudio.AudioTranscription(audio, client)
    
    salvarTexto = SaveFile()
    salvarTexto.SaveTextFile(arquivoTranscricao, transcricao)

    analiseSentimento = AwsServices()
    sentimento = analiseSentimento.SentimentAnalyzer(arquivoTranscricao, comprehend)

    salvarTexto.SaveTextFile(arquivoSentimento, sentimento)

@app.route("/upload", methods=["POST"])    
def carregarArquivo():
    if 'file' not in request.files:
        return jsonify({"error": "Arquivo não enviado"}), 400

    file = request.files.get('file')
    if file.filename == '':
        return jsonify({"error": "Arquivo não selecionado"}), 400
    
    if file.filename.endswith('.mp4'):
        file.save("Arquivos\\Video\\" + file.filename)
        video = f"Arquivos\\Video\\"  + file.filename
        videoReconhecimento = f"Arquivos\\Video\\VideoReconhecimento.mp4"
        audio = f"Arquivos\\Audio\\audio.mp3"
        arquivoTranscricao = f"Arquivos\\Texto\\transcricao.txt"
        arquivoSentimento = f"Arquivos\\Texto\\sentimento.txt"

        DetectarFaceEmocoes(video, videoReconhecimento)
        AnalisarInformacoesVideo(video, audio, arquivoTranscricao, arquivoSentimento)

        return jsonify({"success": "Arquivo carregado com sucesso"}), 200
    else:
        return jsonify({"error": "Erro de Encoding"}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)  
  