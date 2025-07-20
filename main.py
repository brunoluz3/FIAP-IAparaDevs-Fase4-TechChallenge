from dotenv import load_dotenv
from openai import OpenAI
import boto3
import os

from Functions.ConvertVideo import ConvertVideo
from Functions.AwsServices import AwsServices
from Functions.SaveFile import SaveFile
from Functions.Transcription import Transcription


def AnalisarInformacoesVideo():
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    comprehend = boto3.client("comprehend", os.environ.get("region"))

    video = f"Arquivos\\Video\\VideoTesteTranscricao.mp4"
    audio = f"Arquivos\\Audio\\audio.mp3"
    arquivoTranscricao = f"Arquivos\\Texto\\transcricao.txt"
    arquivoSentimento = f"Arquivos\\Texto\\sentimento.txt"

    converterVideo = ConvertVideo()
    converterVideo.ConvertVideoToAudio(video, audio)    

    transcricaoAudio = Transcription()
    transcricao = transcricaoAudio.AudioTranscription(audio, client)
    
    salvarTexto = SaveFile()
    salvarTexto.SaveTextFile(arquivoTranscricao, transcricao)

    analiseSentimento = AwsServices()
    sentimento = analiseSentimento.SentimentAnalyzer(arquivoTranscricao, comprehend)

    salvarTexto.SaveTextFile(arquivoSentimento, sentimento)
    

if __name__ == "__main__":
    AnalisarInformacoesVideo()
    print("Teste")