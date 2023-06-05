
import re
import sys
import numpy as np


def tokenization(sentence):
    # punctuation_re = "[.!?,'\\-:=]"
    punctuation_re = '''!()-[]{};:'"\,<>./?$%^&*_~'''
    my_punct = ['!', '"', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.',
           '/', ':', ';', '<', '=', '>', '?', '[', '\\', ']', '^', '_', 
           '`', '{', '|', '}', '~', '»', '«', '“', '”','-']
    punct_pattern = re.compile("[" + re.escape("".join(my_punct)) + "]")
    url_re = "(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
    hastag_re = "#(\w+)"
    mentions_re = "@(\w+)"
    words_re = "[\w']+"


    clean_sentence = re.sub(punct_pattern,"", sentence)
    clean_sentence = re.sub(url_re,"<URL>", clean_sentence)
    clean_sentence = re.sub(hastag_re,"<HASHTAG>", clean_sentence)
    clean_sentence = re.sub(mentions_re,"<MENTION>", clean_sentence)
    temp= clean_sentence

    if(temp != ' '):
      return temp
    return ''


def load_and_read_corpus(file_path):

    file = open(file_path, 'r', encoding='utf-8')
    data_corpus = file.read()
    data_corpus = data_corpus.split("\n")

    clean_data_corpus = []
    for sentence in data_corpus:
        clean_sentence = tokenization(sentence)
        clean_sentence = clean_sentence.lower().strip()
        if clean_sentence != '':
          clean_sentence = '<s> '+'<s> ' + '<s> ' + clean_sentence + ' </s>'+' </s>'+' </s>'
          clean_data_corpus.append(clean_sentence)
       
    return clean_data_corpus


#TRAINING AND TESTING CORPUS I am selecting the last n lines as test data
def split_train_test_data_corpus(data_corpus,n):
    train_corpus = data_corpus[:-n]
    test_corpus = data_corpus[-n:]
    return train_corpus, test_corpus

def update_dictionary(sen,dict_corpus, unique_words, bigram_corpus, trigram_corpus, quadgram_corpus):
  n=-1
  sentence = sen.split()
  for word in sentence:
      n += 1
      if word not in unique_words:
          # count += 1
          unique_words[word] = 1
          dict_corpus[word] = 1
      else:
          unique_words[word] += 1
          dict_corpus[word] +=1
      #FORMING BI-GRAM
      if n > 0:
          key = sentence[n-1] + ' ' + sentence[n]
          if key not in bigram_corpus:
              bigram_corpus[key] = 1
          else:
              bigram_corpus[key] += 1
              
      #FORMING TRI-GRAM
      if n > 1:
          key = sentence[n-2] + ' ' + sentence[n-1] + ' ' + sentence[n]
          if key not in trigram_corpus:
              trigram_corpus[key] = 1
          else:
              trigram_corpus[key] += 1
      
      #FORMING QUAD-GRAM
      if n>1:
          key =  sentence[n-3] + ' ' + sentence[n-2] + ' ' + sentence[n-1] + ' ' + sentence[n]
          if key not in quadgram_corpus:
              quadgram_corpus[key] = 1
          else:
              quadgram_corpus[key] += 1
  return dict_corpus, unique_words, bigram_corpus, trigram_corpus, quadgram_corpus

#LANGUAGE MODELLING
def build_language_model(train_corpus):
    # count = 0
    unique_words = {}
    dict_corpus = {}
    bigram_corpus = {}
    trigram_corpus = {}
    quadgram_corpus = {}
    for sentence in train_corpus:
        dict_corpus, unique_words, bigram_corpus, trigram_corpus, quadgram_corpus = update_dictionary(sentence,dict_corpus, unique_words, bigram_corpus, trigram_corpus, quadgram_corpus)
        
    return dict_corpus, unique_words, bigram_corpus, trigram_corpus, quadgram_corpus


def Kneyser_Ney_Smoothing(sentence, dict_corpus, bigram_corpus, trigram_corpus, quadgram_corpus):
    t_sentence = sentence.split()
    i = -1
    p = 1.0
    # print(sentence)
    for word in t_sentence:
        i += 1
        if (i > 2):
            # print(i,i-1,i-2,i-3,len(t_sentence))
            unigram_key = t_sentence[i-1]
            bigram_key = t_sentence[i-2] + ' ' + t_sentence[i-1]
            trigram_key = t_sentence[i-3] + ' ' + t_sentence[i-2] + ' ' + t_sentence[i-1]
            quadgram_key = t_sentence[i-3] + ' ' + t_sentence[i-2] + ' ' + t_sentence[i-1] + ' ' +  t_sentence[i]
            start=0
            end=0
            if (quadgram_key in quadgram_corpus.keys()):
                p *= (quadgram_corpus[quadgram_key] - 0.75) / trigram_corpus[trigram_key]                   
            elif (trigram_key in trigram_corpus.keys()):

                for key in trigram_corpus.keys():
                    if key.endswith(word):
                        end += 1
                    if key.startswith(bigram_key):
                        start += 1
                    
                if ((bigram_corpus[bigram_key] > 0) and (start > 0) and (end > 0)):
                    p *= ((0.75/bigram_corpus[bigram_key]) * (start) * ((end) / len(trigram_corpus)))
            elif(bigram_key in bigram_corpus.keys()): 

                for key in bigram_corpus.keys():
                    if key.endswith(word):
                        end += 1
                    if key.startswith(unigram_key):
                        start += 1
                    
                if ((dict_corpus[unigram_key] > 0) and (start > 0) and (end > 0)):
                    p *= ((0.75/dict_corpus[unigram_key]) * (start) * ((end) / len(bigram_corpus)))
            else:
                while (end == 0):
                    start += 1
                    for value in dict_corpus.values():
                      if(value==start):
                        end += value
                p *= (end / len(dict_corpus))
            
    n = len(t_sentence)          
    perplexity_score= (1/p)**(1/float(n))
    return perplexity_score


def find_z_value(words,dict,n,t):
  z=0
  ans=0
  for key in dict.keys():
    if (key.startswith(words)):
      z += 1
  if z>0:
    if n>0:
      if t>0:
        ans = (n/(z*(t+n)))
  return ans

#SMOOTHING TYPE : Witten-Bell Smoothing
def Witten_Bell_Smoothing(sentence, dict_corpus, bigram_corpus, trigram_corpus, quadgram_corpus):

    test_unique_words = {}
    test_dict_corpus = {}
    test_bigram_corpus = {}
    test_trigram_corpus = {}
    test_quadgram_corpus = {}
    test_dict_corpus , test_unique_words , test_bigram_corpus,test_trigram_corpus, test_quadgram_corpus = update_dictionary(sentence,test_dict_corpus , test_unique_words , test_bigram_corpus,test_trigram_corpus, test_quadgram_corpus)
    
    t_sentence = sentence.split()
    n = len(t_sentence)
    dict_value = len(test_unique_words.keys())
    i = -1
    temp_dict = {}
    p = 1
    c_dict = 0
    t_dict = 0
    for word in t_sentence:
        try:
            i = i + 1
            t_dict += 1
            if(word not in temp_dict.keys()):
                c_dict += 1
                temp_dict[word] = 1
                
            if (i > 2):
                unigram_key = t_sentence[i-1]
                bigram_key = t_sentence[i-2] + ' ' + t_sentence[i-1]
                trigram_key = t_sentence[i-3] + ' ' + t_sentence[i-2] + ' ' + t_sentence[i-1]
                quadgram_key = t_sentence[i-3] + ' ' + t_sentence[i-2] + ' ' + t_sentence[i-1] + ' ' +  t_sentence[i]

                t_value = n - t_dict    # remaining words
                n_value = dict_value - c_dict # remainig unique words
                
                if quadgram_key in quadgram_corpus.keys(): 
                    p *= (quadgram_corpus[quadgram_key]) / (n_value + t_value)
                
                else:
                    ans = find_z_value(trigram_key,test_quadgram_corpus,n_value,t_value)
                    if(ans>0):
                      p *= ans
                    if trigram_key in trigram_corpus.keys():
                      p *=  (trigram_corpus[trigram_key]) / (n_value + t_value)
                    else: 
                        ans = find_z_value(bigram_key,test_trigram_corpus,n_value,t_value)
                        if(ans>0):
                            p *= ans
                        if bigram_key in bigram_corpus.keys():
                            p *= (bigram_corpus[bigram_key]) / (n_value + t_value)
                        else:
                            ans = find_z_value(unigram_key,test_bigram_corpus,n_value,t_value)
                            if(ans>0):
                                p *= ans
                            n_1 = 0
                            curr = 0
                            while (n_1 == 0):
                                curr += 1
                                for value in dict_corpus.values():
                                    if curr == value:
                                      n_1 += value
                            l = len(dict_corpus)
                            p *= (n_1/l)
        except:
              continue
    pr_score= (1/p)**(1/float(n))
    return pr_score



def main():
    
    testing_sentence = input("input sentence: ")

    file_path = sys.argv[2]
    data_corpus = load_and_read_corpus(file_path)
    training_corpus, testing_corpus = split_train_test_data_corpus(data_corpus,100)
    dict_corpus, unique_words, bigram_corpus, trigram_corpus, quadgram_corpus = build_language_model(training_corpus)
    smooth = sys.argv[1]
    clean_sentence = tokenization(testing_sentence)
    clean_sentence = clean_sentence.lower().strip()
    if clean_sentence != '':
        clean_sentence = '<s> '+'<s> ' + '<s> ' + clean_sentence + ' </s>'+' </s>'+' </s>'
        pr_score=0.0
        if smooth == "k" :
            pr_score = Kneyser_Ney_Smoothing(clean_sentence, dict_corpus, bigram_corpus, trigram_corpus, quadgram_corpus)
        elif smooth == "w" :
            pr_score = Witten_Bell_Smoothing(clean_sentence, dict_corpus, bigram_corpus, trigram_corpus, quadgram_corpus)
        else:
            print("Warning: choose a valid smoothing !")
        print("perplexity score: ",pr_score)
    else:
        print("perplexity score: ",0.0)
    

main()

