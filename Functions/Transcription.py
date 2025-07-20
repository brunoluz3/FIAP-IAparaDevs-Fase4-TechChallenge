class Transcription:

    """
        Essa funcao sera responsavel por o audio e enviar para a OpenAi para executar a transcricao
    """   
    def AudioTranscription(self, audio_file_path, client):       

        audio_file= open(audio_file_path, "rb")
        transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
        )
        return transcription.text