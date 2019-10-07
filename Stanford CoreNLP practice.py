
# coding: utf-8

# In[1]:


from stanfordcorenlp import StanfordCoreNLP
import json
from pprint import pprint


# In[2]:


nlp = StanfordCoreNLP(r'C:\Users\user\stanfordcorenlp', lang='zh')


# In[ ]:


string = "余知諺大學畢業後選擇在台大資管所繼續鑽研專業領域的技術。"


# In[ ]:


#name-entity recognition 直接呼叫nlp.ner()
string_ner = nlp.ner(string)


# In[ ]:


print(type(string_ner))# list類型
print(type(string_ner[0]))# list中每個元素是一個位元組
print(string_ner)
print("*"*10)


# In[ ]:


#pos-tag 直接呼叫 pos_tag()
string_pos = nlp.pos_tag(string)
print(type(string_pos))# list類型
print(type(string_pos[0]))# list中每個元素是一個位元組
print(string_pos)


# In[4]:


#使用annotate pipeline
text = "I like this chocolate. This chocolate is not good. The chocolate is delicious. Its a very tasty chocolate. This is so bad"
result = nlp.annotate(text,
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
#json.loads()/json.dumps()將JSON編碼的字符串(str)轉換回一個Python數據結構
#如果是文件而非字符串則使用json.load()/json.dump()處理
jsonSentiment = json.loads(result)


# In[5]:


print(jsonSentiment)


# In[6]:


for sentence in jsonSentiment["sentences"]:
    print ( " ".join([word["word"] for word in sentence["tokens"]]) + " => "         + str(sentence["sentimentValue"]) + " = "+ sentence["sentiment"])


# In[7]:


text = "This movie was actually neither that funny, nor super witty. The movie was meh. I liked watching that movie. If I had a choice, I would not watch that movie again."
result2 = nlp.annotate(text,
                   properties={
                       'annotators': 'sentiment, ner, pos',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
jsonSentiment2 = json.loads(result2)


# In[8]:


print(result2)


# In[9]:


print(jsonSentiment2)


# In[ ]:


#lemmatization
for sentence in jsonSentiment2["sentences"]:
    for word in sentence["tokens"]:
        print(word["word"] + " => " + word["lemma"])


# In[ ]:


#POS-tagging
for sentence in jsonSentiment2["sentences"]:
    for word in sentence["tokens"]:
        print (word["word"] + "=>" + word["pos"])


# In[ ]:


#Name-entity recognition
for sentence in jsonSentiment2["sentences"]:
    for word in sentence["tokens"]:
        print (word["word"] + "=>" + word["ner"])


# In[ ]:


for s in jsonSentiment2["sentences"]:
    print("{}: '{}': {} (Sentiment Value) {} (Sentiment)".format(
        s["index"],
        " ".join([t["word"] for t in s["tokens"]]),
        s["sentimentValue"], s["sentiment"]))


# In[ ]:


for sentence in jsonSentiment2["sentences"]:
    print ( " ".join([word["word"] for word in sentence["tokens"]]) + " => "         + str(sentence["sentimentValue"]) + " = "+ sentence["sentiment"])


# In[ ]:


text = "Joshua Brown, 40, was killed in Florida in May when his Tesla failed to "        "differentiate between the side of a turning truck and the sky while "        "operating in autopilot mode."
result3 = nlp.annotate(text,
                   properties={
                       'annotators': 'depparse, sentiment, ner, entitymentions',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
pprint(result3)


# In[ ]:


jsonSentiment3 = json.loads(result3)
#jsonSentiment = json.dumps(result3) 無法轉換??
print(jsonSentiment3)


# In[ ]:


for sentence in jsonSentiment3["sentences"]:
    print ( " ".join([word["word"] for word in sentence["tokens"]]) + " => "         + str(sentence["sentimentValue"]) + " = "+ sentence["sentiment"])

