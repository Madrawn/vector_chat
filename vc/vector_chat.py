import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from datetime import datetime

import tqdm
from extensions.vector_chat.vc.custom_embed import MyEmbeddingFunction
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
        self.current_index = 0
        self.pca = 0

    def clear(self):
        self.current_index = 0
        self.messages = []
        self.indices = []
        for collection in self.client.list_collections():
            self.client.delete_collection(collection.name)

    def set_pca(self, pca):
        self.pca = pca

    def set_distance(self, distance):
        self.distance = distance

    def set_enabled(self, enabled):
        self.enabled = enabled

    def init(self, shared):
        # pass
        ef = MyEmbeddingFunction(model_name="sentence-transformers/gtr-t5-large")

        self.embedding_func = ef

    def add_multiple_messages(self, messages, state):
        for idx, [user, ai] in tqdm.tqdm(enumerate(messages)):
            self.add_message(
                f'{state["name1"] if state["name1"].lower() != "you" else "User"}: {user}\n{state["name2"]}: {ai}' if idx != 0 else f'{state["name2"]}: {ai}',
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
        if self.last_id != unique_id:
            # I remember now, we automatically re embed any message on the first user prompt anyways if we don't find it in the collection
            self.clear() # why did i think this was a good idea?
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
        self.current_index += 1

    def get_chat_context(self, current_message, state, _continue=False):
        chat_context = self._construct_chat_context(
            current_message, self.current_index, state
        )
        messages = self._build_messages(state, chat_context, current_message)
        prompt = self._create_prompt(messages, state, _continue)
        self._log_prompt(prompt)
        return prompt

    def pca_transform(self, embeddings, msg):
        if self.pca == 0:
            return embeddings, msg
        pca_embeddings = self.embedding_func.pca_transform(embeddings["embeddings"] + msg["embeddings"], self.pca)
        return {
            "ids": embeddings["ids"],
            "documents": embeddings["documents"],
            "embeddings": pca_embeddings[:-1],
            "metadatas": embeddings["metadatas"],
        }, pca_embeddings[-1]

    def similar_messages(self, embeddings, message):
        distances = []
        for embedding in embeddings["embeddings"]:
            embedding = embedding.reshape(1,-1)
            message = message.reshape(1,-1)
            print(cosine_similarity(message, embedding))
            distances.append(cosine_similarity(message, embedding)[0])
        embeddings['distances'] = distances
        return embeddings 

    def _construct_chat_context(self, current_message, current_index, state):
        collection = self._get_collection(state)
        embeddings = collection.get(include=["documents", "embeddings", "metadatas"])
        current_message_data = {"ids": [""], "documents": [current_message], "embeddings": self.embedding_func([current_message]), "metadatas": [{"index": current_index}]}
        pca_embeddings, pca_message = self.pca_transform(embeddings, current_message_data)
        similar_messages = self.similar_messages(pca_embeddings, pca_message)
        
        adjusted_similarities = self.calculate_adjusted_similarities(
            current_index, similar_messages, state
        )
        return self.build_chat_context(
            adjusted_similarities,
            state["truncation_length"],
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
        messages.append({"role": "system", "content": chat_context})
        messages.append(
            {"role": "assistent", "content": state["history"]["internal"][-1][1]}
        )
        messages.append({"role": "user", "content": current_message})
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
        with open("ctx.txt", "a", encoding="utf8") as file:
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
            processed_message = msg[1]
            msg_len = get_encoded_length(processed_message)
            if context_len + msg_len + max_new_tokens + buffer <= n_ctx:
                chat_context += self.compose_message_entry(msg)
                context_len += msg_len
            elif msg_len > n_ctx // 3:
                continue
            else:
                break
        chat_context += "\n\n Continuing the conversation:\n"
        return chat_context

    def compose_message_entry(self, msg):
        return "\n".join(
            (
                "(",
                f"{msg[0]-1} messages ago (sim:{msg[-2][0]:.2f}) + (dist:{msg[-4]:.2f}) = res:{msg[-1][0]:.2f}):\n",
                msg[1],
                ")",
            )
        )

    def calculate_adjusted_similarities(self, current_index, similar_messages, state):
        current_index = max(current_index, 1)
        # average_message_length = np.mean(
        #     [
        #         get_encoded_length(message)
        #         for message in similar_messages["documents"][0]
        #     ]
        # )

        # we guess a fair average message length and then set the total message count to
        # the number of messages we can expect to have in the context window.

        # total_message_count = state["max_new_tokens"] // average_message_length

        # Adjust similarity scores based on turn index distance
        adjusted_similarities = []
        for i, (meta, cosine_similarity, text) in enumerate(
            zip(
                similar_messages["metadatas"],
                similar_messages["distances"],
                similar_messages["documents"],
            )
        ):
            # message_length = get_encoded_length(text)
            message_index_distance = abs(self.current_index - meta["index"])
            # normalized_length = message_length / average_message_length
            # relative_msg_length = normalized_length

            relative_index_distance = math.exp(-message_index_distance)
            adjusted_similarity = (cosine_similarity) + (relative_index_distance)

            adjusted_similarities.append(
                (
                    message_index_distance,
                    text,
                    relative_index_distance,
                    0,
                    cosine_similarity,
                    adjusted_similarity,
                )
            )

        # Sort by adjusted similarity
        adjusted_similarities.sort(key=lambda x: x[-1], reverse=True)
        return adjusted_similarities
