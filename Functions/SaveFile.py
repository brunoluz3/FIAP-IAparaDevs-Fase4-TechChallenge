class SaveFile:

    """
        Essa funcao sera responsavel por salvar os arquivos de texto do projeto
    """
    def SaveTextFile(self,name, txt_file):
        text_file = open(name, "w", encoding="utf-8")
        text_file.write(txt_file)
        text_file.close()