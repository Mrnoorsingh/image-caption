#!/usr/bin/env python
# coding: utf-8


from collections import Counter
from string import digits
from keras.preprocessing import sequence
import pickle
import numpy as np



def build_vocab(sentences,threshold):
    new_descrip=[]
    counter=Counter()
    for line in sentences:
        #tokenize sentence into words
        token=line.split()
        #lowercase words in list
        low_word=[word.lower() for word in token]
        #remove digits from the list
        remove_digits = str.maketrans('', '', digits)
        word_list= [i.translate(remove_digits) for i in low_word]
        #remove empty string elements from the list
        remove_space=list(filter(None,word_list))
        #update values in counter
        counter.update(remove_space)
        #make new list of cleaned descriptions
        new_descrip.append(" ".join(remove_space))
    #remove words from vocabulary if frequency is less than given threshold    
    vocabulary=[key for key,value in counter.items() if value>=threshold]    
        
    return vocabulary,new_descrip
        



def max_sequence(description): 
    #find description with maximum length
    max_len=0
    for i in description:
        sent=i.split()
        if len(sent)>max_len:
            max_len=len(sent)  
    return max_len        
            


def data_generator(encoded_img,frame,max_len,batch_size,word2id,clean_desc):
    images=[]
    partial_seq=[]
    next_words=[]
    batch_count=0
    while True:
    
        for idx in range(len(frame)):
            encoded_image=encoded_img[frame[idx][0]]
            encoded_txt=[word2id[text] for text in clean_desc[idx].split()]
            for i in range(1,len(encoded_txt)):
                batch_count+=1
                partial_seq.append(encoded_txt[:i])
                next_words.append(encoded_txt[i])
                images.append(encoded_image)
                
                if batch_count>=batch_size:
                    batch_count=0
                    #returns zero padded sequence
                    partial_seq=sequence.pad_sequences(partial_seq,max_len,padding="post")
                    
                    #one hot encoding for target words
                    hotvector = np.zeros([len(next_words), vocab_size])
                    for i,next_word in enumerate(next_words):
                        hotvector[i,next_word]=1
                        
                    images=np.asarray(images)
                    next_words=np.asarray(hotvector)
                     
                    yield [[images,partial_seq],next_words]    
                    partial_seq=[]
                    next_words=[]
                    images=[]
                
                    
                

