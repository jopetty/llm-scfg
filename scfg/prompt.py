# Generates prompts from grammars and samples

import dataclasses
import json


@dataclasses.dataclass(frozen=True)
class ChatCompletionResponse:
    user_prompt: str
    metadata: dict[str, str] | None = None
    max_new_tokens: int | None = None

    def to_openai_batched_json(self, model: str, custom_id: str) -> str:
        return json.dumps(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": self.user_prompt}],
                    "max_tokens": self.max_new_tokens,
                    "metadata": self.metadata,
                    "store": True if self.metadata else False,
                },
            }
        )


def basic_prompt(
    grammar_str: str,
    sample: str,
):
    prompt: str = f"""You will be presented with a synchronous context-free grammar (SCFG) which defines a mapping between two context-free languages.You will also be presented with a sentence produced by one of the languages defined by the grammar. You task is to use the rules of the grammar to translate the sentence from the source language into the target language.

    A grammar is defined by a set of production rules. Rules come in two forms: non-lexical rules, of the form `A -> <B C, D E>` where all of `A, B, C, D, E` are non-terminal symbols; and lexical rules, of the form `A -> <'a', 'b'>`  where `A` is a non-terminal symbol and `'a'` and `'b'` are terminal symbols (words). The right-hand side of each production rule consists of a pair demarcated by angle brackets. The first element of this pair shows the expansion of the left-hand side in one language, and the second element shows the expansion in the other language. The order of the symbols may differ between the two languages. All grammars are guaranteed to start with a distinguised start symbol `S`. All grammars are defined according to X-bar style rules, intended to model natural language syntax. This means that productions are built are phrases (XP) which produce specifiers (YP) and bar-level projections (XBar); these bar-level projections in turn produce heads (X) and complements (ZP). Certain lexical productions in the grammar produce words which begin with a null symbol '\u2205'; these words are phonetically null and do not appear in the surface forms of either the input or output sentences, though they may be important for the syntactic structure of the sentence. Do not include these null words in your output sentence, though you may need to reason about them to get the correct structure.

    You may use any reasoning strategy you like to solve this task, including identifying the categories of the words in the input sentence, using the grammar to build a parse tree for the input, and then following that derivation using the other language's expansions to produce the output sentence. Feel free to write down intermediate steps in your reasoning.

    You will be evaluated based on the string accuracy of the output sentence, which you should format like the following: `Final answer: <output sentence>`. If you do not end your response with this format, you will be marked as incorrect.

    Here is the synchronous context-free grammar:
    ```
    {grammar_str}
    ```

    Here is the input sentence: `{sample}`.

    Remember to end your response with the format `Final answer: <output sentence>`."""

    return prompt
