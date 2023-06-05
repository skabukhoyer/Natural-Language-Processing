import sys
import pickle
from functools import lru_cache
import numpy as np
import tokenizer as tkzr

##########################################
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import numpy as np
import keras



def prepare_sentence(seq, maxlen):
    # Pads seq and slides windows
    x = []
    y = []
    for i, w in enumerate(seq):
        x_padded = pad_sequences([seq[:i]],
                                maxlen=maxlen - 1,
                                padding='pre')[0]  # Pads before each sequence
        x.append(x_padded)
        y.append(w)
    return x, y



##############################3
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: usage: python3 language_model.py <model file path>")
        exit(-1)

        """

    with open("./ulysses_clean.txt_train.txt",'r') as fp:
        training_data_uly = fp.readlines()[:2000]

    with open("./ulysses_clean.txt_test.txt",'r') as fp:
        test_data_uly = fp.readlines()[:1000]
    """
    
    # smoothing = sys.argv[1]
    neural_model_path  = sys.argv[1]


    filename = neural_model_path.split("/")[-1]

    print(filename)




    user_sentene = input("input sentence: ")

    user_sentene = tkzr.Tokenizer().preprocess_one_sentence(user_sentene)

    print(f"processed senencte = {user_sentene}")

    
    if 'pride_model.h5' == filename:
                    
            with open("./pandp_clean.txt_train.txt",'r') as fp:
                training_data_pride = fp.readlines()

            # with open("./pandp_clean.txt_test.txt",'r') as fp:
            #     test_data_pride = fp.readlines()[:1000]
            # tokenizer

            # Preprocess data
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(training_data_pride)
            vocab = tokenizer.word_index
            seqs = tokenizer.texts_to_sequences(training_data_pride)
            maxlen = max([len(seq) for seq in seqs])

            # load model
            model = keras.models.load_model('./models/pride_model.h5')

            # run on sentence
            tok = tokenizer.texts_to_sequences([user_sentene])[0]
            x_test, y_test = prepare_sentence(tok, maxlen)
            x_test = np.array(x_test)
            y_test = np.array(y_test) - 1  # The word <PAD> does not have a class
            p_pred = model.predict(x_test)
            vocab_inv = {v: k for k, v in vocab.items()}
            log_p_sentence = 0
            for i, prob in enumerate(p_pred):
                word = vocab_inv[y_test[i]+1]  # Index 0 from vocab is reserved to <PAD>
                history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
                prob_word = prob[y_test[i]]
                log_p_sentence += np.log(prob_word)
                # print('P(w={}|h={})={}'.format(word, history, prob_word))
            
            # print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
            num_4_grams = len(user_sentene.split(" ")) - 3
            sent_prob = np.exp(log_p_sentence)
            print(sent_prob)
            

            pass
        ###########
    else:
            
                    
            with open("./ulysses_clean.txt_train.txt",'r') as fp:
                training_data_uly = fp.readlines()[:2000]

            # with open("./pandp_clean.txt_test.txt",'r') as fp:
            #     test_data_pride = fp.readlines()[:1000]
            # tokenizer

            # Preprocess data
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(training_data_uly)
            vocab = tokenizer.word_index
            seqs = tokenizer.texts_to_sequences(training_data_uly)
            maxlen = max([len(seq) for seq in seqs])

            # load model
            model = keras.models.load_model('./models/uly_model.h5')

            # run on sentence
            tok = tokenizer.texts_to_sequences([user_sentene])[0]
            x_test, y_test = prepare_sentence(tok, maxlen)
            x_test = np.array(x_test)
            y_test = np.array(y_test) - 1  # The word <PAD> does not have a class
            p_pred = model.predict(x_test)
            vocab_inv = {v: k for k, v in vocab.items()}
            log_p_sentence = 0
            for i, prob in enumerate(p_pred):
                word = vocab_inv[y_test[i]+1]  # Index 0 from vocab is reserved to <PAD>
                history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
                prob_word = prob[y_test[i]]
                log_p_sentence += np.log(prob_word)
                # print('P(w={}|h={})={}'.format(word, history, prob_word))
            
            # print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
            num_4_grams = len(user_sentene.split(" ")) - 3
            sent_prob = np.exp(log_p_sentence)
            print(sent_prob)
            

