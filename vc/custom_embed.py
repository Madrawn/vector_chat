from chromadb import Documents, EmbeddingFunction, Embeddings
import torch


class MyEmbeddingFunction(EmbeddingFunction):

    tokenizer = None
    model = None

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return [self.embed(doc) for doc in input]

    def embed(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Get embeddings
        with torch.no_grad():
            self.model.generation_config.output_hidden_states = True
            self.model.config.output_hidden_states = True
            outputs = self.model(**inputs)
            embeddings = self.model.model.create_embedding(prompt)
        return embeddings
