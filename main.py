import os
import sqlite3
import time
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('stopwords')

stop_words_slovene = set(stopwords.words("slovene")).union(set(
    ["ter", "nov", "novo", "nova", "zato", "še", "zaradi", "a", "ali", "april", "avgust", "b", "bi", "bil", "bila",
     "bile", "bili", "bilo", "biti",
     "blizu", "bo", "bodo", "bojo", "bolj", "bom", "bomo", "boste", "bova", "boš", "brez", "c", "cel", "cela",
     "celi", "celo", "d", "da", "daleč", "dan", "danes", "datum", "december", "deset", "deseta", "deseti", "deseto",
     "devet", "deveta", "deveti", "deveto", "do", "dober", "dobra", "dobri", "dobro", "dokler", "dol", "dolg",
     "dolga", "dolgi", "dovolj", "drug", "druga", "drugi", "drugo", "dva", "dve", "e", "eden", "en", "ena", "ene",
     "eni", "enkrat", "eno", "etc.", "f", "februar", "g", "g.", "ga", "ga.", "gor", "gospa", "gospod", "h", "halo",
     "i", "idr.", "ii", "iii", "in", "iv", "ix", "iz", "j", "januar", "jaz", "je", "ji", "jih", "jim", "jo",
     "julij", "junij", "jutri", "k", "kadarkoli", "kaj", "kajti", "kako", "kakor", "kamor", "kamorkoli", "kar",
     "karkoli", "katerikoli", "kdaj", "kdo", "kdorkoli", "ker", "ki", "kje", "kjer", "kjerkoli", "ko", "koder",
     "koderkoli", "koga", "komu", "kot", "kratek", "kratka", "kratke", "kratki", "l", "lahka", "lahke", "lahki",
     "lahko", "le", "lep", "lepa", "lepe", "lepi", "lepo", "leto", "m", "maj", "majhen", "majhna", "majhni",
     "malce", "malo", "manj", "marec", "me", "med", "medtem", "mene", "mesec", "mi", "midva", "midve", "mnogo",
     "moj", "moja", "moje", "mora", "morajo", "moram", "moramo", "morate", "moraš", "morem", "mu", "n", "na", "nad",
     "naj", "najina", "najino", "najmanj", "naju", "največ", "nam", "narobe", "nas", "nato", "nazaj", "naš", "naša",
     "naše", "ne", "nedavno", "nedelja", "nek", "neka", "nekaj", "nekatere", "nekateri", "nekatero", "nekdo",
     "neke", "nekega", "neki", "nekje", "neko", "nekoga", "nekoč", "ni", "nikamor", "nikdar", "nikjer", "nikoli",
     "nič", "nje", "njega", "njegov", "njegova", "njegovo", "njej", "njemu", "njen", "njena", "njeno", "nji",
     "njih", "njihov", "njihova", "njihovo", "njiju", "njim", "njo", "njun", "njuna", "njuno", "no", "nocoj",
     "november", "npr.", "o", "ob", "oba", "obe", "oboje", "od", "odprt", "odprta", "odprti", "okoli", "oktober",
     "on", "onadva", "one", "oni", "onidve", "osem", "osma", "osmi", "osmo", "oz.", "p", "pa", "pet", "peta",
     "petek", "peti", "peto", "po", "pod", "pogosto", "poleg", "poln", "polna", "polni", "polno", "ponavadi",
     "ponedeljek", "ponovno", "potem", "povsod", "pozdravljen", "pozdravljeni", "prav", "prava", "prave", "pravi",
     "pravo", "prazen", "prazna", "prazno", "prbl.", "precej", "pred", "prej", "preko", "pri", "pribl.",
     "približno", "primer", "pripravljen", "pripravljena", "pripravljeni", "proti", "prva", "prvi", "prvo", "r",
     "ravno", "redko", "res", "reč", "s", "saj", "sam", "sama", "same", "sami", "samo", "se", "sebe", "sebi",
     "sedaj", "sedem", "sedma", "sedmi", "sedmo", "sem", "september", "seveda", "si", "sicer", "skoraj", "skozi",
     "slab", "smo", "so", "sobota", "spet", "sreda", "srednja", "srednji", "sta", "ste", "stran", "stvar", "sva",
     "t", "ta", "tak", "taka", "take", "taki", "tako", "takoj", "tam", "te", "tebe", "tebi", "tega", "težak",
     "težka", "težki", "težko", "ti", "tista", "tiste", "tisti", "tisto", "tj.", "tja", "to", "toda", "torek",
     "tretja", "tretje", "tretji", "tri", "tu", "tudi", "tukaj", "tvoj", "tvoja", "tvoje", "u", "v", "vaju", "vam",
     "vas", "vaš", "vaša", "vaše", "ve", "vedno", "velik", "velika", "veliki", "veliko", "vendar", "ves", "več",
     "vi", "vidva", "vii", "viii", "visok", "visoka", "visoke", "visoki", "vsa", "vsaj", "vsak", "vsaka", "vsakdo",
     "vsake", "vsaki", "vsakomur", "vse", "vsega", "vsi", "vso", "včasih", "včeraj", "x", "z", "za", "zadaj",
     "zadnji", "zakaj", "zaprta", "zaprti", "zaprto", "zdaj", "zelo", "zunaj", "č", "če", "često", "četrta",
     "četrtek", "četrti", "četrto", "čez", "čigav", "š", "šest", "šesta", "šesti", "šesto", "štiri", "ž", "že",
     "svoj", "jesti", "imeti", "\u0161e", "iti", "kak", "www", "km", "eur", "pač", "del", "kljub", "šele", "prek",
     "preko", "znova", "morda", "kateri", "katero", "katera", "ampak", "lahek", "lahka", "lahko", "morati", "torej",
     "(", ")", "--", ";", ".", ",", "/", "-", "!", "?", "'", ":"]))

conn = sqlite3.connect('inverted-index.db')
cur = conn.cursor()

path_to_file = "C:/Users/miham/Documents/Faks/IEPS/PA3/PA3-data/evem.gov.si/evem.gov.si.4.html"
root = "C:\\Work\\Magisterij_1_leto\\2.semester\\ekstrakcijaSplet\\nal3\\PA3-data"
stop_words_english = stopwords.words('english')

query=["evidenca","nov"]
docs=[]
print(f'Results  for a query: "{query}"\n')
start_time = time.time()
candidates=[]
for subdir, dirs, files in os.walk(root):
    for file in files:

        path = os.path.join(subdir, file)
        domain = os.path.basename(os.path.normpath(subdir))
        site = file
        addr = domain + '/' + site

        docName=path.split("\\")[-1] #names of documents
       # print(fileName)
        with open(path, encoding='utf8') as fp:
            frequency=0
            soup = BeautifulSoup(fp, 'html.parser')

            for script in soup(["script", "style"]):
                script.extract()  # remove script tags, style tags

            # get text, tokenize text, and remove stop words
            soup2=soup.__copy__()
            [s.extract() for s in soup2(['style', 'script', '[document]', 'head', 'title'])]
            tokens_orig = word_tokenize(soup2.get_text(separator=" "))
            print(tokens_orig)
            text = soup.get_text()
            text = text.lower()
            textORG=soup.get_text(separator=" ")
            tokens = word_tokenize(text)
            #print(tokens)
            tokens = [t for t in tokens if not t in stop_words_slovene]
            tokens = [t for t in tokens if not t in stop_words_english]
            # find spans of tokens in original text
           # print(tokens)
            spans = [] ##indexi
            ix = 0
            for token in tokens:
                if token in query:
                    frequency+=1
                ix = text.find(token, ix)
                spans.append(ix)
                ix += len(token)
            print(frequency)
            if(frequency>0):
                candidates.append([frequency,docName,"noSnippet"])
        break
endTime=time.time()-start_time
print(f"Found query in {candidates} pages")
print("Result found in {:0.2f} ms\n".format(endTime*1000))
nsnipets=2
Fspace=13
Dspace=40
Sspace=50
begin = f"{'Frequencies':<{Fspace}}{'Document':<{Dspace}}"
for i in range(nsnipets):
    snipp = f"Snippet {i}"
    begin += f"{snipp:<{Sspace}}"
print(begin,"\n","-" * (Fspace + Dspace * nsnipets +Sspace))

snipets=["hey","grem"]
docs=["vlsaads.fslo","test2.js"]
frekvence=[4,9]
for i in range(len(candidates)):
    line = f"{candidates[i][0]:<{Fspace}}{candidates[i][1]:{Dspace}}"
    for i in range(len(snipets)):
        line+=f"{snipets[i]:<{Sspace}}"
    print(line)