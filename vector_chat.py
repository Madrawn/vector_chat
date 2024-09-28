import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from some_vector_db import VectorDB  # Replace with actual vector DB library
from some_embedding_model import EmbeddingModel  # Replace with actual embedding model library



class ChatInterface:
    def set_enabled(self, enabled):
        self.enabled = enabled
        if enabled:
            self.vector_db = VectorDB()
            self.embedding_model = EmbeddingModel()
            self.messages = []
            self.indices = []

    def add_message(self, message, index):
        embedding = self.embedding_model.encode(message)
        self.vector_db.insert(embedding, {'message': message, 'index': index})
        self.messages.append(message)
        self.indices.append(index)

    def get_chat_context(self, current_message, current_index):
        current_embedding = self.embedding_model.encode(current_message)
        similar_messages = self.vector_db.query(current_embedding, top_k=10)
        
        # Adjust similarity scores based on turn index distance
        adjusted_similarities = []
        for msg in similar_messages:
            distance = abs(current_index - msg['index'])
            adjusted_similarity = msg['similarity'] / (1 + distance)
            adjusted_similarities.append((msg['message'], adjusted_similarity))
        
        # Sort by adjusted similarity
        adjusted_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Construct chat context
        chat_context = [msg[0] for msg in adjusted_similarities[:10]]
        return chat_context
    
# # Example usage
chat_interface = ChatInterface()
# chat_interface.add_message("Hello, how can I help you?", 0)
# chat_interface.add_message("Can you tell me about your services?", 1)
# chat_interface.add_message("Sure, we offer a variety of services including...", 2)

# current_message = "What services do you offer?"
# current_index = 3
# chat_context = chat_interface.get_chat_context(current_message, current_index)
# print(chat_context)
    
# Extension that modifies the chat history before it is used
# def _apply_history_modifier_extensions(history):
#     """
#     Modify the chat history using the given extensions.

#     Args:
#         history (list): The chat history.

#     Returns:
#         list: The modified chat history.
#     """
#     for extension, _ in iterator():
#         if hasattr(extension, "history_modifier"):
#             history = getattr(extension, "history_modifier")(history)

#     return history

