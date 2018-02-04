'''

This is used for testing purposes only, might be deleted later on.


'''
import csv
words=[]
with open('consolidated.csv') as csvfile:
	reader=csv.DictReader(csvfile)
	for row in reader:
		if int(row['valid'])==1:
			words.append([row['tweet'], row['corrected']])
#print(data)
import random
#print(words)
'''data=random.sample(words,10500)
data_test=data[:int(0.0477*10500)]
data_train=data[int(0.0477*10500):]
print(len(data_test))
print len(data_train)
'''
tweet_train=[]
corrected_train=[]
for x in words:
	tweet_train.append(x[0])
	corrected_train.append(x[1])
#print(corrected_train)
from keras.preprocessing.text import text_to_word_sequence
sentences=[]
sentences_output=[]
for x in tweet_train:
	sentences.append(text_to_word_sequence(x))
for y in corrected_train:
	#print(y)
	sentences_output.append(text_to_word_sequence(y))
#print(t)
import gensim
#t=gensim.models.build_vocab(sentences_output)
model = gensim.models.Word2Vec(sentences,min_count=1)
#print(t)
data=random.sample(words,10500)
data_test=data[:int(0.0477*10500)]
data_train=data[int(0.0477*10500):]
data_train_tweets=[]
data_train_corrected=[]
for x in data_train:
	data_train_tweets.append(text_to_word_sequence(x[0]))
	data_train_corrected.append(text_to_word_sequence(x[1]))
inputs=[]
for x in data_train_tweets:
	temp=[]	
	for y in x:
		temp.append(model.wv[y])
	inputs.append(temp)
print(len(inputs))
	
