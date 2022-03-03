# Script generico para utilizar rouge

from rouge import Rouge
from rouge import FilesRouge
#pip install rouge

if __name__ == "__main__":

     hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

     reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

     rouge = Rouge()
     files_rouge = FilesRouge()
     scores = files_rouge.get_scores("rouge_data/generado.txt", "rouge_data/ideal.txt")

     print (scores)