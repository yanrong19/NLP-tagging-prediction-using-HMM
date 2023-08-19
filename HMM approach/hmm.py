# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv("twitter_train.txt", sep="\t+", engine="python", header=None)
df2 = pd.read_csv("twitter_train.txt", sep="\t+", engine="python", skip_blank_lines=False, header=None)
df.columns = ["token","tag"]
df2.columns = ["token","tag"]
tt = pd.read_csv("twitter_tags.txt", sep="\t", header=None)
tt.columns = ["tag"]


# In[2]:


#Q4a)
transition_count_dict = {}

for i in range(len(df["tag"])-1):
    tag1 = df2["tag"][i] #assign tag 1
    tag2 = df2["tag"][i+1] #assign tag 2
    if tag1 in transition_count_dict.keys():
        if tag2 in transition_count_dict[tag1].keys():
            transition_count_dict[tag1][tag2] += 1 #tag1 -> tag2 already present in dictionary
        else:
            transition_count_dict[tag1][tag2] = 1 #tag1 -> tag2 not present in dictionary
    else:
        transition_count_dict[tag1] = {}
        transition_count_dict[tag1][tag2] = 1 #tag1 -> tag2 not present in dictionary


# In[3]:


#Q4a)
#convert count to probability of each transition from tag1 to tag2
transition_prob_dict = {}
δ = 0.01
num_tags = 25 + 1 #25 unique tags +  1 blank line represented as start/end

for tag1, tag2s in transition_count_dict.items():
    total_count = sum(tag2s.values()) + δ*(num_tags + 1) #sum total counts of tag1
    transition_prob = {}
    for tag2, count in tag2s.items():
        transition_prob[tag2] = (count + δ)/ total_count #count(tag1 -> tag2)/count(tag1)
    transition_prob["unknown"] = δ/total_count  #store probability of unseen transition of tags
    transition_prob_dict[tag1] = transition_prob


# In[4]:


#Q4a)
import json
with open('trans_probs.txt', 'w') as convert_file:
     convert_file.write(json.dumps(transition_prob_dict))
        
with open('trans_probs2.txt', 'w') as convert_file:
     convert_file.write(json.dumps(transition_prob_dict))


# In[5]:


#Q4a)
output_count_dict = {}

for tag in tt["tag"]:
    token_dict = {} #create a dictionary for each tag to store count of tokens that correspond to that tag
    for token in df[df["tag"]==tag]["token"]: #filter tokens that have this tag and iterate through these tokens
        if token in token_dict.keys():
            token_dict[token] += 1
        else:
            token_dict[token] = 1
    output_count_dict[tag] = token_dict #store this tag into the output dictionary and move on to next tag (if any)


# In[6]:


#Q4a)
#convert count to probability of each token appearing given tag
output_prob_dict = {}
δ = 0.01 # possible values: 0.01, 0.1, 1, 10
num_words = df['token'].nunique() #number of unique tokens in training data

for tag in output_count_dict.keys(): #iterate through all tags
    #num_words = df[df["tag"]==tag]['token'].nunique()
    total_prob = 0
    count_yj = sum(output_count_dict[tag].values()) + δ*(num_words + 1) #count total number of times this tag appeared in training data
    token_dict={} #create a dictionary for each tag to store probability of tokens given tag
    for token in output_count_dict[tag].keys():
        token_dict[token] = (output_count_dict[tag][token] + δ)/count_yj #convert count to probability
        total_prob +=token_dict[token]
    token_dict["unknown"] = δ/count_yj #store probability of unseen emission of tokens (including both seen and unseen tokens in training data)
    output_prob_dict[tag] = token_dict #store the probability distributions into the output dictionary


# In[7]:


#Q4a)
import json
with open('output_probs.txt', 'w') as convert_file:
     convert_file.write(json.dumps(output_prob_dict))

with open('output_probs2.txt', 'w') as convert_file:
     convert_file.write(json.dumps(output_prob_dict))


# 5a) To improve the POS tagger, 2 methods were implemented to better handle unseen words
# 
# Firstly, we implemented rule-based tagging by identifying tokens that have specific patterns. 
# This can allow us to better map unseen tokens based on their patterns and their corresponding association with a certain tag. 
# We let the probability of a tag y emitting an unseen token with such patterns be equal to the probability of observing tag y emitting tokens with such patterns given all tokens with such patterns in the training data. 
# This helps to better capture the correct tags of unseen tokens in the test set.
# 
# Patterns captured:
# 1. Tokens that start with '@USER' are usually tagged with '@'
# 2. Tokens that contain 'http://' are tagged with 'U'
# 3. Tokens that contain '!' are usually tagged with ',' 
# 4. Tokens that contain ' ' ' are usually tagged with ',' , 'L', or 'V'
# 5. Tokens that start with '#' are usually tagged with '#'
# 6. Tokens that end with 'ing' are usually tagged with 'V'
# 7. Tokens that are in title form are usually tagged with '^', 'N', 'O' or 'V'
# 
# Secondly, some unseen token may not neccessarily be unseen. They may appear in different forms but was not captured by the model in the training data. 
# For example, "Apple" and "apple" may refer to the same token associated with the same tag but the model sees these 2 tokens as different even though they mean the same thing. 
# To tackle this issue, tokens are converted into 3 forms, lower case form, upper case form and title form. 
# The model iterates through these 4 forms (including the original) to check if the token is actually captured in the output probability data. 
# As such, these words are not considered unseen and the probability of associating them with the correct tag is greater.
# 
# Other improvements (not implemented) may involve using a more complex model such as a second-order Hidden Markov model which captures the transition probabilites from 2 prior instances.
# Other work can include structure learning by calculating the Bayesian Information Criterion (BIC) and determining which hidden Markov model would best illustrate the data.
# Perhaps different variations of the hidden Markov model may yield higher accuracies. 
# 
# Additionally, we can assign weights to tags based on their degree of "openness". 
# For example, some tags such as "P" (prepositions) have certain fixed words and hence are quite closed. Tags such as "E" (Expressions) do not have fixed words and are highly variable, hence are quite open. 
# Given an unseen token, they likely belong to a tag which is more open, and thus we can assign higher probabilities to these tags.

# In[8]:


def word_patterns(tag, token): #rule-based tagging to improve tagging of unseen tokens in test data
    if (token[:5] == "@USER") & (tag == "@"):
        return 770/771 # P(token that start with @USER, tag = @) / P(token that start with @USER)
    elif ("!" in token) & (tag==","):
        return 373/379 # P(token that has !, tag = ,) / P(token that has !)
    elif ("http://" in token) & (tag=="U"):
        return 1 # P(token that has http://, tag = U) / P(token that has http://)
    elif ("'" in token) & (tag in [",","L","V"]):
        if tag==",":
            return 34/366 # P(token that has ', tag = ,) / P(token that has ')
        elif tag=="L":
            return 181/366 # P(token that has ', tag = L) / P(token that has ')
        else:
            return 96/366 # P(token that has ', tag = V) / P(token that has ')
    elif (token[:1] == "#") & (tag=="#"):
        return 167/257 # P(token that starts with #, tag = #) / P(token that starts with #)
    elif (token.istitle()) & (tag in ["^","N","O","V"]):
        if tag == "^":
            return 663/2380 # P(token is in title form, tag = ^) / P(token is in title form)
        elif tag == "N":
            return 389/2380 # P(token is in title form, tag = N) / P(token is in title form)
        elif tag == "O":
            return 356/2380 # P(token is in title form, tag = O) / P(token is in title form)
        else:
            return 273/2380 # P(token is in title form, tag = V) / P(token is in title form)
    elif (token[-3:] == "ing") & (tag=="V"):
        return 235/344 # P(token ends with "ing", tag = V) / P(token ends with "ing")
    else:
        return 0.0001 #still unknown tokens, assign arbitrarily low probability


# In[9]:


#Q4b)
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    import numpy as np
    import pandas as pd
    import json
    trans_prob_dict = json.load(open(in_trans_probs_filename))
    output_prob_dict = json.load(open(in_output_probs_filename))
    test = pd.read_csv(in_test_filename, sep="\t+", engine="python", header=None, skip_blank_lines=False)
    test.columns = ["token"]
    twitter_tags = pd.read_csv(in_tags_filename, sep="\t+", engine="python", header=None)
    twitter_tags.columns = ["tags"]
    hidden_states = twitter_tags["tags"].to_dict() #assign each tag a number to allow easy manipulation of the matrices
    
    def viterbi(output_prob_dict, transition_prob_dict, test, twitter_tags): #define main function of viterbi
        pi = np.zeros(shape=(len(test), len(hidden_states))) #initialise pi matrix to store max probability for each token
        BP = np.zeros(shape=(len(test), len(hidden_states))) #initialise BP matrix to track the tag corresponding to the max probability for each token
        start = test["token"][0] #first token
        for v in hidden_states.keys():
            try:
                transition = transition_prob_dict["null"][hidden_states[v]] #initial probabilities: transition from blank line (represented as null) to tag v
            except:
                transition = transition_prob_dict["null"]["unknown"] #no such initial probabilities: assign to unknown
            try:
                emission = output_prob_dict[hidden_states[v]][start] #emission probability of first token
            except:
                emission = output_prob_dict[hidden_states[v]]["unknown"] #no such emission probability: assign to unknown

            pi[0][v] = transition * emission
            BP[0][v] = -1
        for k in range(1, len(test)):
            for v in hidden_states.keys():
                max_prob = 0
                max_prob_hs = None
                for u in hidden_states.keys():
                    try:
                        transition = transition_prob_dict[hidden_states[u]][hidden_states[v]] #transition probabilities from tag u to tag v
                    except:
                        transition = transition_prob_dict[hidden_states[u]]["unknown"] #unseen transitions: assign to unknown
                    try:
                        emission = output_prob_dict[hidden_states[v]][test["token"][k]] #emission probability of token
                    except:
                        emission = output_prob_dict[hidden_states[v]]["unknown"]#unseen token asign to unknown
                    prob = pi[k-1][u] * transition * emission
                    if prob > max_prob:
                        max_prob = prob #track the highest probability
                        max_prob_hs = u #track the tag corresponding to the highest probability
                pi[k][v] = max_prob
                BP[k][v] = max_prob_hs
        maxProb = 0
        finalBP = None
        for v in hidden_states.keys():
            try:
                transition = transition_prob_dict[hidden_states[v]]["null"] #transition probabilities from tag v to STOP (represented by null)
            except:
                transition = transition_prob_dict[hidden_states[v]]["unknown"] #unseen transition probability from tag v to STOP, assign to unknown
            prob = pi[len(test)-1][v] * transition
            if prob > maxProb:
                maxProb = prob
                finalBP = v
        return BP, finalBP
    
    def backtrack(BP, finalBP):
        #iterate through the backpointers starting from the end, to backtrack the tags
        arr = [hidden_states[finalBP]]
        for i in range(len(BP)-1,0,-1):
            finalBP = BP[i][int(finalBP)]
            tag = hidden_states[finalBP]
            arr.append(tag)
        return arr[::-1]
    
    #perform viterbi for every tweet
    tweet_start_indexes = test[test["token"].isnull()].index.values #get the start index of every tweet
    start = 0
    pred = [] #store the predictions
    for index in tweet_start_indexes:
        tweet = test.iloc[start:index].reset_index()
        BP, finalBP = viterbi(output_prob_dict, trans_prob_dict, tweet, twitter_tags)
        bt = backtrack(BP, finalBP)
        pred.extend(bt)
        start = index + 1
    pred_df = pd.DataFrame(pred)
    pred_df.to_csv(out_predictions_filename, sep='\t', index=False, header=None)

#Q5b)
def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    import numpy as np
    import pandas as pd
    import json
    trans_prob_dict = json.load(open(in_trans_probs_filename))
    output_prob_dict = json.load(open(in_output_probs_filename))
    test = pd.read_csv(in_test_filename, sep="\t+", engine="python", header=None, skip_blank_lines=False)
    test.columns = ["token"]
    twitter_tags = pd.read_csv(in_tags_filename, sep="\t+", engine="python", header=None)
    twitter_tags.columns = ["tags"]
    hidden_states = twitter_tags["tags"].to_dict() #assign each tag a number to allow easy manipulation of the matrices
    def word_patterns(tag, token): #rule-based tagging to improve tagging of unseen tokens in test data
        if (token[:5] == "@USER") & (tag == "@"):
            return 770/771 # P(token that start with @USER, tag = @) / P(token that start with @USER)
        elif ("!" in token) & (tag==","):
            return 373/379 # P(token that has !, tag = ,) / P(token that has !)
        elif ("http://" in token) & (tag=="U"):
            return 1 # P(token that has http://, tag = U) / P(token that has http://)
        elif ("'" in token) & (tag in [",","L","V"]):
            if tag==",":
                return 34/366 # P(token that has ', tag = ,) / P(token that has ')
            elif tag=="L":
                return 181/366 # P(token that has ', tag = L) / P(token that has ')
            else:
                return 96/366 # P(token that has ', tag = V) / P(token that has ')
        elif (token[:1] == "#") & (tag=="#"):
            return 167/257 # P(token that starts with #, tag = #) / P(token that starts with #)
        elif (token.istitle()) & (tag in ["^","N","O","V"]):
            if tag == "^":
                return 663/2380 # P(token is in title form, tag = ^) / P(token is in title form)
            elif tag == "N":
                return 389/2380 # P(token is in title form, tag = N) / P(token is in title form)
            elif tag == "O":
                return 356/2380 # P(token is in title form, tag = O) / P(token is in title form)
            else:
                return 273/2380 # P(token is in title form, tag = V) / P(token is in title form)
        elif (token[-3:] == "ing") & (tag=="V"):
            return 235/344 # P(token ends with "ing", tag = V) / P(token ends with "ing")
        else:
            return 0.0001 #still unknown tokens, assign arbitrarily low probability
    
    def cap(a): # a function which returns [original token, token in lowercase form, token in uppercase form, token in title form] given a token
        return [a, a.lower(), a.title(), a.upper()]
    
    def viterbi(output_prob_dict, transition_prob_dict, test, twitter_tags): #define main function of viterbi
        pi = np.zeros(shape=(len(test), len(hidden_states))) #initialise pi matrix to store max probability for each token
        BP = np.zeros(shape=(len(test), len(hidden_states))) #initialise BP matrix to track the tag corresponding to the max probability for each token
        start = test["token"][0] #first token
        for v in hidden_states.keys():
            try:
                transition = transition_prob_dict["null"][hidden_states[v]] #initial probabilities: transition from blank line (represented as null) to tag v
            except:
                transition = transition_prob_dict["null"]["unknown"] #no such initial probabilities: assign to unknown
            try:
                i = 0
                for t in cap(start): #iterate though 4 forms of the token
                    try:
                        emission = output_prob_dict[hidden_states[v]][t] #check if any form matches a seen token in training data
                        break #if matched then break out of this loop
                    except:
                        i += 1
                if i==4: #no matches to seen tokens
                    raise Exception("Still unseen")
            except:
                #check for token patterns in unseen token and recalibrate the emission probability using word_patterns function
                emission = output_prob_dict[hidden_states[v]]["unknown"] * word_patterns(hidden_states[v], start)
            pi[0][v] = transition * emission
            BP[0][v] = -1
        for k in range(1, len(test)):
            token = test["token"][k]
            for v in hidden_states.keys():
                max_prob = 0
                max_prob_hs = None
                for u in hidden_states.keys():
                    try:
                        transition = transition_prob_dict[hidden_states[u]][hidden_states[v]] #transition probabilities from tag u to tag v
                    except:
                        transition = transition_prob_dict[hidden_states[u]]["unknown"] #unseen transition probability
                    try:
                        i = 0
                        for t in cap(token): #iterate though 4 forms of the token
                            try:
                                emission = output_prob_dict[hidden_states[v]][t] #check if any form matches a seen token in training data
                                break #if matched then break out of this loop
                            except:
                                i += 1
                        if i==4: #no matches to seen tokens
                            raise Exception("Still unseen")
                    except:
                        #check for token patterns in unseen token and recalibrate the emission probability using word_patterns function
                        emission = output_prob_dict[hidden_states[v]]["unknown"] * word_patterns(hidden_states[v], token)
                    prob = pi[k-1][u] * transition * emission
                    if prob > max_prob:
                        max_prob = prob 
                        max_prob_hs = u
                pi[k][v] = max_prob
                BP[k][v] = max_prob_hs
        maxProb = 0
        finalBP = None
        for v in hidden_states.keys():
            try:
                transition = transition_prob_dict[hidden_states[v]]["null"] #transition probabilities from tag v to STOP (represented by null)
            except:
                transition = transition_prob_dict[hidden_states[v]]["unknown"] #unseen transition probabilities from tag v to STOP 
            prob = pi[len(test)-1][v] * transition
            if prob > maxProb:
                maxProb = prob
                finalBP = v
        return BP, finalBP
    
    def backtrack(BP, finalBP):
        #iterate through the backpointers starting from the end, to backtrack the tags
        arr = [hidden_states[finalBP]]
        for i in range(len(BP)-1,0,-1):
            finalBP = BP[i][int(finalBP)]
            tag = hidden_states[finalBP]
            arr.append(tag)
        return arr[::-1]
    
    #perform viterbi for every tweet
    tweet_start_indexes = test[test["token"].isnull()].index.values #get the start index of every tweet
    start = 0
    pred = [] #store the predictions
    for index in tweet_start_indexes:
        tweet = test.iloc[start:index].reset_index()
        start = index + 1
        BP, finalBP = viterbi(output_prob_dict, trans_prob_dict, tweet, twitter_tags)
        bt = backtrack(BP, finalBP)
        pred.extend(bt)
    pred_df = pd.DataFrame(pred)
    pred_df.to_csv(out_predictions_filename, sep='\t', index=False, header=None)

def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '.' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

#     naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
#     naive_prediction_filename = f'{ddir}/naive_predictions.txt'
#     naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
#     correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
#     print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

#     naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
#     naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
#     correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
#     print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    run()


# Q4c) Viterbi1 Accuracy = 75.5%
# Q5c) Viterbi2 Accuracy = 82.4%

# %%
