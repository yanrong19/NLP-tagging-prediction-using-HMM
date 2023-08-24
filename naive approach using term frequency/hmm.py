#import training data set and twitter tags
import pandas as pd
df = pd.read_csv("twitter_train.txt", sep="\t+", engine="python", header=None)
df.columns = ["token","tag"]
tt = pd.read_csv("twitter_tags.txt", sep="\t", header=None)
tt.columns = ["tags"]

#for each tag, count number of times each token appear
output_count_dict = {}

for tag in tt["tags"]:
    token_dict = {} #create a dictionary for each tag to store count of tokens that correspond to that tag
    for token in df[df["tag"]==tag]["token"]: #filter tokens that have this tag and iterate through these tokens
        if token in token_dict.keys():
            token_dict[token] += 1
        else:
            token_dict[token] = 1
    output_count_dict[tag] = token_dict #store this tag into the output dictionary and move on to next tag (if any)

#convert count to probability of each token appearing given tag
num_words = df['token'].nunique() #number of unique tokens in training data
δ = 1 # possible values: 0.01, 0.1, 1, 10
output_prob_dict = {}

for tag in output_count_dict.keys(): #iterate through all tags
    count_yj = sum(output_count_dict[tag].values()) + δ*(num_words + 1) #count total number of times this tag appeared in training data
    token_dict={} #create a dictionary for each tag to store probability of tokens given tag
    for token in output_count_dict[tag].keys():
        token_dict[token] = (output_count_dict[tag][token] + δ)/count_yj #convert count to probability
    output_prob_dict[tag] = token_dict #store the probability distributions into the output dictionary

#write to txt file
import json
with open('naive_output_probs.txt', 'w') as convert_file:
     convert_file.write(json.dumps(output_prob_dict))

#model design
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    import pandas as pd
    import json
    prob_dict = json.load(open(in_output_probs_filename))
    tdnt = pd.read_csv(in_test_filename, sep="\t+", engine="python", header=None)
    tdnt.columns = ["token"]
    pred_df = pd.DataFrame() #create new dataframe to store the predicted tags
    for token in tdnt["token"]: #iterate through each token in test data
        curr_highest_prob = 0 #keep track of highest probability value
        curr_highest_tag = None #keep track of tag corresponding to highest probability value
        for tag in prob_dict.keys(): #iterate through all tags
            try: #try to see if test token has this tag assigned to it in training data
                prob = prob_dict[tag][token] #if present, then store the probability value P(x = w | y = j)
                if prob > curr_highest_prob: #check if this probability value is higher than current highest
                    curr_highest_prob = prob #if so then assign new highest probability value
                    curr_highest_tag = tag #assign new tag corresponding to new highest probability value
            except: #this token does not have this tag assigned to it in the training data
                pass
        pred_df = pd.concat([pred_df, pd.DataFrame([curr_highest_tag])]) #append the tag corresponding to the highest probability value to ans_df
    pred_df.to_csv(out_prediction_filename, sep='\t', index=False, header=None) #store ans_df in the out_prediction_filename

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    import pandas as pd
    import json
    prob_dict = json.load(open(in_output_probs_filename))
    df = pd.read_csv(in_train_filename, sep="\t+", engine="python", header=None)
    df.columns = ["token","tag"]
    tdnt = pd.read_csv(in_test_filename, sep="\t+", engine="python", header=None)
    tdnt.columns = ["token"]
    pred_df = pd.DataFrame() #create new dataframe to store the predicted tags
    for token in tdnt["token"]: #iterate through each token in training data
        curr_highest_prob = 0 #keep track of highest probability value
        curr_highest_tag = None #keep track of tag corresponding to highest probability valu
        for tag in prob_dict.keys(): #iterate through all tags
            try: #try to see if test token has this tag assigned to it in training data
                prob = prob_dict[tag][token] #if present, then store the probability value P(x = w | y = j)
                prob_yj = len(df[df["tag"]==tag]) / len(df["tag"]) #P(y = j)
                prob_xw = len(df[df["token"]==token]) / len(df["token"]) #P(x = w)
                prob = (prob * prob_yj) / prob_xw #convert to P(y = j | x = w) = P(x = w | y = j) * P(y = j)/P(x = w) using Bayes' Rule
                if prob > curr_highest_prob: #check if this probability value is higher than current highest
                    curr_highest_prob = prob #if so then assign new highest probability value
                    curr_highest_tag = tag #assign new tag corresponding to new highest probability value
            except: #this token does not have this tag assigned to it in the training data
                pass
        pred_df = pd.concat([pred_df, pd.DataFrame([curr_highest_tag])]) #append the tag corresponding to the highest probability value to ans_df
    pred_df.to_csv(out_prediction_filename, sep='\t', index=False, header=None) #store ans_df in the out_prediction_filename

def evaluate(in_prediction_filename, in_answer_filename):
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
    ddir = '.' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

if __name__ == '__main__':
    run()

# P(y = j | x = w) is the probability of tag j being associated with token w given token w appearing. <br>
# Using Baye's Rule, P(y = j | x = w) = P(x = w | y = j) * P(y = j)/P(x = w) where P(x = w | y = j) is given by naive_output_probs.txt <br>
# We keep track of the tag j that gives the highest value of P(y = j | x = w) given a token w.
# 
# δ = 0.01 <br>
# Naive prediction accuracy: 893/1378 = 0.6480406386066764 <br>
# Naive prediction2 accuracy: 913/1378 = 0.6625544267053701
# 
# δ = 0.1 <br>
# Naive prediction accuracy: 900/1378 = 0.6531204644412192 <br>
# Naive prediction2 accuracy: 912/1378 = 0.6618287373004355
# 
# δ = 1 <br>
# Naive prediction accuracy: 908/1378 = 0.6589259796806967 <br>
# Naive prediction2 accuracy: 909/1378 = 0.6596516690856313
# 
# δ = 10 <br>
# Naive prediction accuracy: 908/1378 = 0.6589259796806967 <br>
# Naive prediction2 accuracy: 895/1378 = 0.6494920174165457
# 
# Based on the accuracy values, δ = 1 gave the highest overall accuracy for both models. Hence, δ = 1 was chosen.
# 
# For δ = 1, Naive prediction accuracy = 908/1378 = 0.6589259796806967
# 
# For δ = 1, Naive prediction2 accuracy = 909/1378 = 0.6596516690856313
