from moviepy import VideoFileClip

class ConvertVideo:

    """
        Essa funcao sera responsavel por converter o video para um audio e salvar o arquivo no mesmo diretorio do video
    """
    def ConvertVideoToAudio(self, video_path, audio_name):
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_name)
