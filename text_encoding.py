from dapi_lib import models


text = "Hello World!"

# Initialize model
#weights = models.RobertaWeights.BASE
#model = models.roberta(weights)
model, weights = models.get('roberta', models.RobertaWeights.BASE)

model.eval()

# Initialize tokenizer
tokenizer = weights.transforms()

# Tokenize
encoded_input = tokenizer(text)

# Apply model
output = model(**encoded_input)

# Show number of params and hidden state shape
print(weights.meta['params'], output.last_hidden_state.shape)
