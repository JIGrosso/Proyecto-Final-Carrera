# Script generico para utilizar rouge

from rouge import Rouge
from rouge import FilesRouge
#pip install rouge

if __name__ == "__main__":

     rouge = Rouge()
     files_rouge = FilesRouge()
     scores = files_rouge.get_scores("rouge_data/generado.txt", "rouge_data/ideal.txt")

     print (scores)