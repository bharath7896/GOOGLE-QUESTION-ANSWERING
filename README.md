# GOOGLE-QUESTION-ANSWERING
Discovering the Multi-Intent behind the given query title and body and ranking the question answering with multiple feature metrics
## QUESTION-ANSWER INTENT RANKING BY KEYWORD FILTERING AND INTENT UNDERSTANDING
* Generally we understand a query or a setence by breaking down the structural phrases and then understanding the semantics
* for a query to be more specific the 'question-word','POS sequence' and 'entity' are three core parameters to understand its intent more spcifically 
1. We are going to extract the structural differences and the semantic differences seperately and then combine them to get the perfect order of the clasification result 
2. We are going to consider the key words i,e q-word, verbs , entities of properly framed questions
3. for improper structural sentences and queries with non-q words we are going to convert them into a proper question and then we are going to apply the features on those sentences.
4. Thus for every well framed and well structured sentence we are going to rank the categories based on how frequently the keywords have occured in the corresponding categories and at which position of the sentence and how the intent filters that we are going to apply are matching the inferences.
