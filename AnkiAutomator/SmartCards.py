from textblob import TextBlob
import nltk
import spacy
import streamlit as st

nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



subprocess.run(cmd)
print("Working")
nlp = spacy.load("en_core_web_sm")
import PyPDF2

import genanki
import numpy as np




def scrapePDF(pdfFile, pageNums: list = None, returnAll: bool = False):

    try: 
        allText = '' 
        pdfDict = {}


        reader = PyPDF2.PdfReader(pdfFile)
        numberPages = len(reader.pages)

        st.text('Successfully read PDF!')
        st.text('')

        if pageNums == None: 
            lower_page, higher_page = 0, numberPages
        
        elif (isinstance(pageNums, list) or isinstance(pageNums, tuple)) and len(pageNums) == 2: 
            lower_page, higher_page = int(pageNums[0] -1), int(pageNums[1])
        elif (isinstance(pageNums, list) or isinstance(pageNums, tuple)) and len(pageNums) ==1: 
            lower_page, higher_page = 0 , int(pageNums[1])

        elif isinstance(pageNums, str) or isinstance(pageNums, int) or isinstance(pageNums, float): 
            lower_page, higher_page = 0,  int(pageNums)
            try: 
                for page_num in range(lower_page, higher_page):
                    page = reader.pages[page_num]
                    pageContent= page.extract_text() 
                    pdfDict[f'Page {page_num+1}'] = pageContent
                    allText += pageContent
            except Exception as e:
                print()


        else: 
    
            raise TypeError("Invalid Page Number Input")
        


        for page_num in range(lower_page, higher_page):
                page = reader.pages[page_num]
                pageContent= page.extract_text() 
                pdfDict[f'Page {page_num+1}'] = pageContent
                allText += pageContent
    

        if returnAll == True: 
            return (allText, pdfDict) 
        else: 
            return (allText)
    except Exception as e: 
        st.text(e)



  

















stringFunction = [
"""

def stringSplitter(strng: str, splitOn: str,  x_occurence = 1 ,randChoice: list =[1,2]) -> list:
  randSplits = False
  if randChoice != None: 
    randSplits = True
  
  splitCopy = strng.split(splitOn)
  strList = []
  for inx, strings in enumerate(splitCopy): 
    if randSplits == True: 
      x_occurence = np.random.choice(randChoice)
    if ((inx+1) % x_occurence == 0):
      #print(' '.join(splitCopy[:inx +1]).strip())
      strList.append(' '.join(splitCopy[:inx +1]).strip())
      splitCopy = splitCopy[inx+1:];
  strList = [i for i in strList if len(i) >= 3]
  return strList

"""
]

def createFlashCards(article, deckName: str, attachName= 'ankiDeck', splitOn = '.',   x_occurence = 1, randChoice = [1,2],changeRate = 0.10, verbose = False):

  articleList = []
  nounList = []
  questionAnswerDict = {}
  articles = article.split(splitOn)

  for inx, n in enumerate(articles): 
    articleList.append(n)
    nounList= (
         [str(x) for x in [nouns for nouns in nlp(n).noun_chunks] if str(x).lower() not in ['he', 'him', 'his', 'her','she', 'they', 'them', 'who','how', 'it']]
    )




    randNum = np.random.choice([0,1])
    if randNum == 1:
      nounList = nounList[int(len(nounList) * (1 -changeRate)):]
    else: 
      nounList = nounList[:-int(len(nounList) * (1 -changeRate))]
                                                                          # Let's also get some subjects in here. 
    try: 
      nounList  +=  [str(tokens) for tokens in nlp(n) if (tokens.dep_ == "nsubj") and (str(tokens).lower() not in ['he', 'him', 'his', 'her','she', 'they', 'them', 'who','how', 'it'])]

    except Exception as e: 
      print('')
   
    for nouns in nounList: 
      n = n.replace(nouns, '___', 1)
      n = n.replace('\n', '')
      n = n.strip()


    if (n.count(' ') >= 3) and len( str(',   '.join([x.strip('\n') for x in nounList]))) >= 2 :
      questionAnswerDict[n] = str(',   '.join([x.strip('\n') for x in nounList]))

        



    ankiModel = genanki.Model(
      2042686211,
      'Simple Model',
      fields=[
          {'name': 'Question'},
          {'name': 'Answer'},
      ],
      templates=[
          {
              'name': 'Card 1',
              'qfmt': '{{Question}}',
              'afmt': '{{FrontSide}}<hr id="answer"><b>{{Answer}}<b>',
          },
      ])


    ankiDeck = genanki.Deck(
        1724897887,
        deckName)
  

  inx = 0
  for keyz, vals in  questionAnswerDict.items():
    aNote = genanki.Note(
        model=ankiModel, fields=[keyz , vals ]
    )
    ankiDeck.add_note(aNote)

  # Output anki file in desired folder
  
  ankiCards = f'{attachName}.apkg'
  genanki.Package(ankiDeck).write_to_file(
      f'{attachName}.apkg')
  print('\n\nGenerated Anki Deck!')

  return ankiCards;



st.title('SmartCards :robot_face:')


optionsMenu = ["About", "Main App"]

optionsChoice = st.sidebar.selectbox("Options", optionsMenu)
if optionsChoice == "About":
    st.header("About The App:")
    st.markdown("""This app uses Natural Lanaguage Processing/AI to automate the generation of
                Anki Flashcards. Currently **there are two ways to use it**: either by directly pasting
                text, or by uploading a PDF. Future plans involves using AI to summarize text and better
                identifying the optimal words to "mask" in each question. Options such as scraping
                websites, word documents, and videos will also be included.""")

    st.text('')

st.text('')
st.text('')
test_text = st.text_area("""**Drop Text In Here and I'll Turn Them Into Flashcards**""")

deckName=  st.text_input('**Please Enter The Name Of The Deck:**')
uploadedFile = None;

with st.sidebar.header('**Upload A PDF To Turn Into Flashcards**'):
    uploadedFile = st.sidebar.file_uploader("Please Upload a PDF file", type=["pdf"])
    if uploadedFile != None:
        reader = PyPDF2.PdfReader(uploadedFile)
        numberPages = len(reader.pages)


        pageVals = np.arange(start = 1, stop = numberPages+1, step = 1)
        lower_bound, upper_bound = ( st.select_slider('Select Which Pages To Turn Into Flashcards', options = pageVals,
                            value = (min(pageVals), max(pageVals))))

        





if uploadedFile != None and deckName != '': 
    try: 
        st.markdown('**Converting Uploaded PDF to Cards! This may take a few minutes.**')
        exportedCards = createFlashCards(article = scrapePDF(pdfFile = uploadedFile, pageNums = [lower_bound, upper_bound]) , deckName = deckName)
        with open(exportedCards, 'rb') as ankiFile:
            st.text('')
            st.text('')
            st.download_button(
            label='Download Cards (PDF)',
            data= ankiFile,
            file_name= exportedCards,
            mime='text/csv')

    except Exception as e: 
        st.text(e)



if test_text != '' and deckName != '':
    try: 
        exportedCards = createFlashCards(article = test_text, deckName = deckName)
        
        with open(exportedCards, 'rb') as ankiFile:
            st.text('')
            st.text('')
            st.text('')

            st.markdown('**Converting Uploaded to Cards! This may take a few minutes.**')
            st.text("Note: no cards may be generated if small text size");
            st.text('')
            st.text('')
            st.download_button(
            label='Download Cards',
            data= ankiFile,
            file_name= exportedCards,
            mime='text/csv')

    except Exception as e:
        st.text(e)




