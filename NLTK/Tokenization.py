import nltk
paragraph = '''Artificial Intelligence refers to the intelligence of machines. This is in contrast to the natural 
               intelligence of\nhumans and animals. With Artificial Intelligence, machines perform functions such as 
               learning, planning, reasoning and\nproblem-solving. Most noteworthy, Artificial Intelligence is the simulation 
               of human intelligence by machines.\nIt is probably the fastest-growing development in the World of technology 
               and innovation. Furthermore, many experts believe\nAI could solve major challenges and crisis situations.'''

# Tokenization  sentences
sentences = nltk.sent_tokenize(paragraph)

# Tokenizing words
words = nltk.word_tokenize(paragraph)               