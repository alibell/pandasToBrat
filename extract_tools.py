import pandas as pd
import numpy as np
import re

##
# extract_tools : functions used in extract functionnality
##


###             ###
### TOKENIZERS  ###
###             ###

#
# Tokenizers : Functions that cut string to tokens and calculate pos tags
#
# How to write a Tokenizer ?
#   input : should be a sentence
#   output : should be an array [[token, start offset, end offset, pos tag], ...]
#

def default_tokenizer (x):

    '''
        default_tokenizer
        The minimal tokenizer, cut when a blanc space or an new line exists.

        input : str, sentence
        output : array, [[token, start offset, end offset], ...]
    '''

    split_catacters = " |\n"
        
    tokens = pd.DataFrame(re.split(split_catacters, x)).rename(columns = {0:'token'})
    
    # Splitted word size
    tokens["size"] = tokens["token"].apply(len)
    
    # Cutting caracter size after the spitted word
    tokens["empty_space"] = 1
    
    # Cum sum of empty space and size
    temp_cum_sum = tokens.cumsum()
    
    tokens["start_offset"] = (temp_cum_sum["size"]-tokens["size"])+temp_cum_sum["empty_space"]-1
    tokens["end_offset"] = tokens["size"]+tokens["start_offset"]
    tokens["pos_tag"] = "O"
    
    tokens_list = tokens[["token", "start_offset","end_offset", "pos_tag"]].values.tolist()
    
    return (tokens_list)

def spacy_tokenizer(nlp):
    
    '''
        Function that generate a tokenizer from Spacy object.
        
        input : spacy nlp function
        output : tokenizer function for export function of pandasToBrat
    '''
    
    def _spacy_tokenizer(x):
        
        tokens_data = pd.DataFrame(nlp(x))
        tokens_data["tokens"] = tokens_data[0].apply(lambda x: x.text)
        tokens_data["size"] = tokens_data["tokens"].str.len()
        tokens_data["start_offset"] = tokens_data[0].apply(lambda x: x.idx)
        tokens_data["end_offset"] = tokens_data["start_offset"]+tokens_data["size"]
        tokens_data["pos"] = tokens_data[0].apply(lambda x: x.pos_)        
        
        output_list = tokens_data[["tokens", "start_offset", "end_offset", "pos"]].values.tolist()
        
        return(output_list)
    
    return(_spacy_tokenizer)