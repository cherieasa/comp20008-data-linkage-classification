# Code written by Terada Asavapakuna 1012869

# import libraries
import pandas as pd
import random
import string
import nltk
#nltk.download('stopwords') 
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

# define stopwords list we aim to cleanse
stop_words = stopwords.words('english')

# read csv file to df
amazon_df = pd.read_csv("amazon.csv")
google_df = pd.read_csv("google.csv")

# extract title
a_list_title = (amazon_df.title).tolist()
g_list_name = (google_df.name).tolist()

# remove punctuation from title
def remove_punct(title_list):
    # remove punct and append to stripped_title
    table = str.maketrans('', '', string.punctuation)
    stripped_title = [w.translate(table) for w in title_list]
    
    return(stripped_title)

# remove stop words
def stop_word_removal(stripped_title):
    no_stop_word = []
    for sentence_list in stripped_title:
            text_tokens = word_tokenize(sentence_list)
            tokens_no_stopword = [word for word in text_tokens if not word in stopwords.words()]
            joined = ' '.join(tokens_no_stopword)
            no_stop_word.append(joined)

    return no_stop_word

a_stop_remove = stop_word_removal(a_list_title)
g_stop_remove = stop_word_removal(g_list_name)
a_stripped_title = remove_punct(a_stop_remove)
g_stripped_title = remove_punct(g_stop_remove)

# function to create a dictionary with title sentence as key and ids as (list) value 
# if multiple stemmed sentence identical - add id as a list in value 
# function for each amazon and google dict separately

def create_title_id_dict(list_title, list_ids):
    dup_dict = dict()
    for i in range(len(list_title)):
        # test if there is a duplicate
        if list_title[i] in dup_dict:
            # append the new id to the existing array at this slot
            dup_dict[list_title[i]].extend([list_ids[i]])
        else:
            # create a new array in this slot
            dup_dict[list_title[i]] = [list_ids[i]]
            
    return dup_dict

a_dict = create_title_id_dict(a_stop_remove, amazon_df.idAmazon.tolist())
g_dict = create_title_id_dict(g_stop_remove, google_df.id.tolist())

# ----------------------- blocking method -----------------------

# function to return best match and score as a list
def sim_score(word_to_compare, compare_to_list):
    best_match = process.extract(word_to_compare,compare_to_list,scorer = fuzz.partial_ratio, limit = 1)
    
    for word,value in best_match:
        best_match_word = word
        best_match_score = value
    
    return [best_match_word, best_match_score]

# dictionary of blocks
block_dict = {}

# loop through each value google dictionary
for title, ids in g_dict.items():
    
    # if no blocks are created yet in dictionary - make new
    if (len(block_dict)) == 0:
        # add key as words (as block id for now) and value as first id
        block_dict[title] = ids

    # calculate string similarity of each google title to block, if similar then add to same id
    else:
        key_list = list(block_dict.keys())
        
        # check similarity of blocking key of all blocks currently in block_dict with current title
        best_match = sim_score(title, key_list)
        highest = best_match[1]
        best_key = best_match[0]
        
        # check if the highest value is high enough to be added to a block
        if (highest > 85):
            
            # keep existing ids that were in the block
            original = []
            original.extend(block_dict[best_key])
            
            # add to original existing ids
            original.extend(ids)
            # reset new key value
            block_dict[best_key] = original
            
        # not similar enough then create a new block
        else: 
            block_dict[title] = ids

block_keys = list(block_dict.keys())

# loop through every amazon item with existing blocks made by using google title
for title, ids in a_dict.items():

    # compare title with all keys in block
    best_block_match = sim_score(title, block_keys)[0]
    
    # keep existing ids that were in the block
    original_ids = []

    original_ids.extend(block_dict[best_block_match])
    
    # add to original existing ids
    original_ids.extend(ids)
    
    # reset new ids value for that key
    block_dict[best_block_match] = original_ids
    
# create unique blocking keys 
characters = (string.ascii_letters+ string.digits)

def generate_unique_key():
    unique_key = ''.join(random.sample(characters, 2))
    # prevent chance of same key
    if (unique_key in block_dict.keys()):
        return generate_unique_key()
    else:
        return unique_key   

# modify dictionary by changing blocking_key (title) -> unique block_id
# loop in list of values creating a separate dictionary (amazon/google) of itemid (key), blockid (value)
a_block_dict = {}
g_block_dict = {}
count = 0

for block_key, list_ids in block_dict.items():
    # change block_key for each key
    new_key = generate_unique_key()
    
    # loop through list_ids
    for a_g_ids in list_ids:
        # check if it belongs in amazon dictionary - ids dont start with h
        if (a_g_ids[0] != "h"):
            a_block_dict[a_g_ids] = new_key
        
        # if it starts with h (link) - its a google id
        elif (a_g_ids[0] == "h"):
             g_block_dict[a_g_ids] = new_key
             
        # to check if there is a block missing
        else:
            print("not allocated")
            
    count += 1
    
# create dataframe 
task1b_amazon = {"block_key" : list(a_block_dict.values()), "product_id": list(a_block_dict.keys())}
task1b_google = {"block_key" : list(g_block_dict.values()), "product_id": list(g_block_dict.keys())}

task1b_amazon_df = pd.DataFrame(task1b_amazon) 
task1b_google_df = pd.DataFrame(task1b_google) 

# output to csv file 
task1b_amazon_df.to_csv('amazon_blocks.csv', index=False)        
task1b_google_df.to_csv('google_blocks.csv', index=False)   

  










