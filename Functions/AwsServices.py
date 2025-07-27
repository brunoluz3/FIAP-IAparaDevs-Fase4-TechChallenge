class AwsServices:

    """
        Essa funcao sera responsavel por abrir o texto e enviar para analise do serviço comprehend da aws
    """
    def SentimentAnalyzer (self, txt_file, aws_comprehend):
        with open(txt_file, "r", encoding="utf-8") as file:
            text = file.read()

        sentiment_response = aws_comprehend.detect_sentiment(
            Text = text[:4000],
            LanguageCode="pt"
        )

        sentiment = sentiment_response["Sentiment"]
        return sentiment