import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import torch
import functools
import extensions.vector_chat.vc.custom_embed as custom_embed
from chromadb.utils import embedding_functions
import hashlib
from datetime import datetime
from modules.chat import replace_character_names, get_generation_prompt
from jinja2.sandbox import ImmutableSandboxedEnvironment
from functools import partial

from modules.text_generation import get_encoded_length

# Copied from the Transformers library
jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)


def remove_extra_bos(prompt):
    for bos_token in ["<s>", "<|startoftext|>", "<BOS_TOKEN>", "<|endoftext|>"]:
        while prompt.startswith(bos_token):
            prompt = prompt[len(bos_token) :]

    return prompt


def make_prompt(
    messages,
    state,
    renderer,
    instruct_renderer,
    instruction_template,
    _continue=False,
    impersonate=False,
):
    if state["mode"] == "chat-instruct" and _continue:
        prompt = renderer(messages=messages[:-1])
    else:
        prompt = renderer(messages=messages)

    if state["mode"] == "chat-instruct":
        outer_messages = []
        if state["custom_system_message"].strip() != "":
            outer_messages.append(
                {"role": "system", "content": state["custom_system_message"]}
            )

        prompt = remove_extra_bos(prompt)
        command = state["chat-instruct_command"]
        command = command.replace(
            "<|character|>", state["name2"] if not impersonate else state["name1"]
        )
        command = command.replace("<|prompt|>", prompt)
        command = replace_character_names(command, state["name1"], state["name2"])

        if _continue:
            prefix = get_generation_prompt(
                renderer, impersonate=impersonate, strip_trailing_spaces=False
            )[0]
            prefix += messages[-1]["content"]
        else:
            prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]

        outer_messages.append({"role": "user", "content": command})
        outer_messages.append({"role": "assistant", "content": prefix})

        prompt = instruction_template.render(messages=outer_messages)
        suffix = get_generation_prompt(instruct_renderer, impersonate=False)[1]
        if len(suffix) > 0:
            prompt = prompt[: -len(suffix)]

    else:
        if _continue:
            suffix = get_generation_prompt(renderer, impersonate=impersonate)[1]
            if len(suffix) > 0:
                prompt = prompt[: -len(suffix)]
        else:
            prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
            prompt += prefix

    prompt = remove_extra_bos(prompt)
    return prompt


class ChatInterface:

    collection: chromadb.Collection = None

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path="./extensions/vector_chat/chroma_db",
        )

        self.embedding_func = None
        self.messages = []
        self.indices = []
        self.enabled = False
        self.distance = "l2"
        self.last_id = None

    def clear(self):
        self.messages = []
        self.indices = []
        for collection in self.client.list_collections():
            self.client.delete_collection(collection.name)

    def set_distance(self, distance):
        self.distance = distance

    def set_enabled(self, enabled):
        self.enabled = enabled

    def init(self, shared):
        # pass
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/gtr-t5-large"
        )

        self.embedding_func = ef

    def add_multiple_messages(self, messages, state):
        for idx, [user, ai] in enumerate(messages):
            self.add_message(
                f'{state["name1"] if state["name1"].lower() != "you" else "User"}: {user}\n{state["name2"]}: {ai}',
                idx,
                state["unique_id"],
            )

    def refresh_db(self):
        collections_info = f">>{self.last_id}<<\n"
        collections_info += "\n".join(
            f"{collection.name}: {collection.count()}"
            for collection in self.client.list_collections()
        )
        return collections_info

    def add_message(self, message: str, index: int, unique_id: str):
        self.last_id = unique_id
        collection: chromadb.Collection = self.client.get_or_create_collection(
            unique_id,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"},
        )

        # Generate a hash of the message
        message_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()

        # Check if the message hash is already in the collection
        existing_ids = collection.get(ids=[message_hash])
        if existing_ids["ids"]:
            print(f"Message with hash {message_hash} already exists in the collection.")
            return

        collection.add(
            ids=[message_hash], documents=[message], metadatas={"index": index}
        )
        self.messages.append(message)
        self.indices.append(index)

    def get_chat_context(self, current_message, current_index, state, _continue=False):
        chat_context = self._construct_chat_context(
            current_message, current_index, state
        )
        messages = self._build_messages(state, chat_context, current_message)
        prompt = self._create_prompt(messages, state, _continue)
        self._log_prompt(prompt)
        return prompt

    def _construct_chat_context(self, current_message, current_index, state):
        collection = self._get_collection(state)
        similar_messages = collection.query(query_texts=current_message, n_results=1000)
        adjusted_similarities = self.calculate_adjusted_similarities(
            current_index, similar_messages
        )
        return self.build_chat_context(
            adjusted_similarities,
            state["n_ctx"],
            state["max_new_tokens"],
            current_message,
        )

    def _get_collection(self, state):
        return self.client.get_or_create_collection(
            state["unique_id"],
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": self.distance},
        )

    def _build_messages(self, state, chat_context, current_message):
        messages = []
        messages.append(
            {"role": "assistent", "content": state["history"]["visible"][-1][1]}
        )
        messages.append({"role": "user", "content": current_message})
        messages.append({"role": "system", "content": chat_context})
        return messages

    def _get_chat_template_str(self, state):
        chat_template_str = state["chat_template_str"]
        if state["mode"] != "instruct":
            chat_template_str = replace_character_names(
                chat_template_str, state["name1"], state["name2"]
            )
        return chat_template_str

    def _create_prompt(self, messages, state, _continue):
        instruction_template = jinja_env.from_string(state["instruction_template_str"])
        chat_template = jinja_env.from_string(self._get_chat_template_str(state))
        renderer = self._get_renderer(state, instruction_template, chat_template)
        
        return make_prompt(
            messages,
            state,
            renderer=renderer,
            instruct_renderer=instruction_template.render,
            instruction_template=instruction_template,
            _continue=_continue,
            impersonate=False,
        )

    def _get_renderer(self, state, instruction_template, chat_template):
        if state["mode"] == "instruct":
            return partial(
                instruction_template.render,
                builtin_tools=None,
                tools=None,
                tools_in_user_message=False,
                add_generation_prompt=False,
            )
        else:
            return partial(
                chat_template.render,
                add_generation_prompt=False,
                name1=state["name1"],
                name2=state["name2"],
                user_bio=replace_character_names(
                    state["user_bio"], state["name1"], state["name2"]
                ),
            )

    def _log_prompt(self, prompt):
        with open("ctx.txt", "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write("\n".join(["#"] * 10))
            file.write(f"Timestamp: {timestamp}\n")
            file.write(prompt + "\n")

    def build_chat_context(
        self, adjusted_similarities, n_ctx, max_new_tokens, current_message, buffer=100
    ):
        if len(adjusted_similarities) == 0:
            return ""
        chat_context = "You recall these messages:\n"
        context_len = get_encoded_length(current_message) + buffer
        for msg in adjusted_similarities:  # Process messages from most recent to oldest
            msg_len = get_encoded_length(msg[1])
            if context_len + msg_len + max_new_tokens + buffer <= n_ctx:
                chat_context += "\n".join(
                    (
                        "(",
                        f"{msg[0]-1} messages ago (initial_similarity:{msg[-2]:.2f} * adjusted_length:{msg[-3]:.2f} * adjusted_distance_factor:{msg[-4]:.2f} = adjusted_similarity:{msg[-1]:.2f}):\n",
                        msg[1],
                        ")",
                    )
                )
                context_len += msg_len
            else:
                break
        chat_context += "\n\n Continuing the conversation:\n"
        return chat_context

    def build_chat_context2(
        self, adjusted_similarities, n_ctx, max_new_tokens, current_message
    ):
        chat_context = "You recall these messages:\n"
        chat_context += "".join(
            [
                "\n".join(
                    (
                        "(",
                        f"{msg[0]-1} messages ago (initial_similarity:{msg[-2]:.2f} * adjusted_length:{msg[-3]:.2f} * adjusted_distance_factor:{msg[-4]:.2f} = adjusted_similarity:{msg[-1]:.2f}):\n",
                        msg[1],
                        ")",
                    )
                )
                for msg in adjusted_similarities[:10]
            ]
        )
        chat_context += "\n\n Continuing the conversation:\n"
        return chat_context

    def calculate_adjusted_similarities(self, current_index, similar_messages):
        average_message_length = np.mean(
            [len(message) for message in similar_messages["documents"][0]]
        )
        total_message_count = len(similar_messages["ids"][0])

        # Adjust similarity scores based on turn index distance
        adjusted_similarities = []
        for i, (meta, initial_similarity, text) in enumerate(
            zip(
                similar_messages["metadatas"][0],
                similar_messages["distances"][0],
                similar_messages["documents"][0],
            )
        ):
            # meta = similar_messages["metadatas"][0][i]
            # initial_similarity = similar_messages["distances"][0][i]
            # text = similar_messages["documents"][0][i]
            message_length = len(text)
            message_index_distance = abs(current_index - meta["index"])
            normalized_index_distance = message_index_distance / (
                max(total_message_count - 1, 1)
            )
            normalized_length = message_length / average_message_length
            adjusted_length = message_length * normalized_length

            adjusted_distance_factor = math.exp(
                -normalized_index_distance / (2 * current_index)
            )
            adjusted_similarity = (
                initial_similarity * adjusted_length * adjusted_distance_factor
            )

            adjusted_similarities.append(
                (
                    message_index_distance,
                    text,
                    adjusted_distance_factor,
                    adjusted_length,
                    initial_similarity,
                    adjusted_similarity,
                )
            )

        # Sort by adjusted similarity
        adjusted_similarities.sort(key=lambda x: x[-1], reverse=True)
        return adjusted_similarities


# # Example usage
# chat_interface = ChatInterface()
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
