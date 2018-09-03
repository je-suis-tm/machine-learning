# coding: utf-8

# In[1]:

#this is simply an optimization for naive bayes
#the default sklearn package is so damnnnn slow
#so i use a lil memoization technique to improve its time complexity
#for details of naive bayes, plz refer to the following link
# https://github.com/tattooday/machine-learning/blob/master/naive%20bayes.ipynb

import pandas as pd
import re
import os
os.chdir('h:/')


# In[2]:

#convert text into a list of words
def text2list(text,lower=False):
    
    temp=text if lower==False else text.lower()
    regex=re.findall('\w*',temp)
    output=list(filter(lambda x: x!='',regex))
    
    return output


# In[3]:

#build up vocabulary for all words in our training data
def get_vocabulary(output,stopword):
    
    vocabulary=sorted(list(set(output)))
    
    for i in vocabulary:
        if i in stopword:
            vocabulary.remove(i)
        
    return vocabulary


# In[4]:

#calculate conditional probability with laplace smoothing
def multivariate_calc_prob(word,x_train,y_train,classification):
    

    num=list(y_train).count(classification)
    
    temp=[i.count(word) for i in x_train[y_train==classification]]
    freq=len(temp)-temp.count(0)
    
    if freq!=0:
        p=freq/num
        
    else:
        p=(freq+1)/(num+2)

    return p


#memoization
#calculate every conditional probability in our training dataset
#and we store these probabilities in a local folder
#so everytime we wanna make forecast
#we do not need to caculate these probabilities again
def multivariate_store_prob(sample,stopword):
    
    temp=[]
    for i in sample['title']:
        temp.append(text2list(i,lower=True))

    sample['word']=temp
    
    phi_y0=list(sample['key']).count(0)/len(sample)
    phi_y1=1-phi_y0
    
    df=pd.DataFrame()
    
    df['entire training sample']=[phi_y0,phi_y1]
    
    for i in sample['word']:
        for j in i:
            if j not in stopword:
                px_y0=multivariate_calc_prob(j,                                               
                                             sample['word'],                                               
                                             sample['key'],0)
                px_y1=multivariate_calc_prob(j,                                               
                                             sample['word'],                                               
                                             sample['key'],1)
                
                df[j]=[px_y0,px_y1]
            else:
                pass

    return df


#when we make forecast on several words
#we intend to find the conditional probability in our local file first
#if it doesnt exist, we would get a keyerror
#we just ignore it and take 1/2 as the laplace smoothed probability
#we create a new column in dataframe to store the forecast
def multivariate_forecast(df,dic,stopword):
    
    temp=[]
    for i in df['title']:
        temp.append(text2list(i,lower=True))

    df['word']=temp
    forecast=[]
    
    phi_y0=dic['entire training sample'][0]
    phi_y1=dic['entire training sample'][1]

    
    for j in df['word']:
        px_y0,px_y1=1,1

        for k in j:
            if k not in stopword:
                try:
                    px_y0*=dic[k][0]
                    px_y1*=dic[k][1]
                except KeyError:
                    px_y0*=1/2
                    px_y1*=1/2
            else:
                pass
        
        
        py0_x=px_y0*phi_y0
        py1_x=px_y1*phi_y1

        p=0 if py0_x>py1_x else 1
        forecast.append(p)
        
    df['forecast']=forecast
    
    return df


# In[5]:

#this is the stopword i have thought of
#anything i am missing?
stopword=['i','we','our','my','me','you',           
          'your','to','ours','yours','him','his',           
          'he','her','hers','she','they','their',           
          'theirs','them','in','s','of','for',           
          'u', 'the', 'with', 'a', 'us', 'and',           
          'on', 'from','as', 'over', 'after',            
          'is', 'are', 'by','at','above','beyond',          
          'after','before','within','around','about',           
          'up','will','would','be']


#when this file is run as a main
#we could add some new training datasets into our csv file first
#it would automatically update our vocabulary and conditional probabilities
#next time we import this file
#we would get more accurate forecast
def main():
    
    #i use latin-1 cuz utf_8_sig cannot encode some characters
    #some symbols could turn out to be a mess
    #as we only care about words and we have regex to fix it
    #we do not need to worry too much about it
    #and dic is shorter form for dictionary:P
    df=pd.read_csv('training dataset.csv',encoding='latin-1')
    dic=multivariate_store_prob(df,stopword)
    dic.to_csv('local data.csv',index=False)
    
    
    #this is how we use this file to forecast
    
    """
    df=pd.read_csv('testing data.csv',encoding='latin-1')
    new=multivariate_forecast(df,dic,stopword)
    new.to_csv('testing data.csv',encoding='utf_8_sig')
    """


if __name__ == "__main__":
    main()


