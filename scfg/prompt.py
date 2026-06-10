# Generates prompts from grammars and samples

import dataclasses
import json
from typing import Any


@dataclasses.dataclass(frozen=True)
class ChatCompletionResponse:
    user_prompt: str
    metadata: dict[str, str] | None = None
    max_completion_tokens: int | None = None
    n: int = 1

    def to_openai_batched_json(self, model: str, custom_id: str) -> str:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": self.user_prompt}],
            "max_completion_tokens": self.max_completion_tokens,
            "n": self.n,
        }
        obj: dict[str, Any] = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        if self.metadata is not None:
            body["metadata"] = self.metadata
            body["store"] = True
        return json.dumps(obj=obj)


def basic_prompt(
    grammar_str: str,
    sample: str,
    agreement_metadata: dict | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
):
    agreement_note = ""
    if agreement_metadata and agreement_metadata.get("enabled"):
        agreement_note = (
            "\n\n"
            "    Some lexical items in this grammar inflect for grammatical "
            "features such as person, number, and sometimes gender. Compact "
            "lexical entries may summarize an entire paradigm after the lemma "
            "pair, for example `V -> <'lemma_a', 'lemma_b'> "
            "(3.sg.fem=<'form_a', 'form_b'>; ...)`. Use those annotations to "
            "determine which surface form is required in the target language."
        )

    examples_note = ""
    if few_shot_examples:
        example_lines = [
            (
                f"    {index}. Source: `{example['input']}`\n"
                f"       Target: `{example['output']}`"
            )
            for index, example in enumerate(few_shot_examples, start=1)
        ]
        examples_note = (
            "\n\n"
            "    Here are example translations produced by this same grammar:\n"
            + "\n".join(example_lines)
            + "\n\n"
        )

    prompt: str = (
        "You will be presented with a synchronous context-free grammar (SCFG) "
        "which defines a mapping between two context-free languages. You will "
        "also be presented with a sentence produced by one of the languages "
        "defined by the grammar. Your task is to use the rules of the grammar "
        "to translate the sentence from the source language into the target "
        "language.\n\n"
        "    A grammar is defined by a set of production rules. Rules come in "
        "two forms: non-lexical rules, of the form `A -> <B C, C B>` where "
        "all of `A, B, C` are non-terminal symbols; and lexical rules, "
        "of the form `A -> <'a', 'b'>` where `A` is a non-terminal symbol and "
        "`'a'` and `'b'` are terminal symbols (words). The right-hand side of "
        "each production rule consists of a pair demarcated by angle "
        "brackets. The first element of this pair shows the expansion of the "
        "left-hand side in one language, and the second element shows the "
        "expansion in the other language. The order of the symbols may differ "
        "between the two languages. All grammars are guaranteed to start with "
        "a distinguished start symbol `S`. All grammars are defined according "
        "to X-bar style rules, intended to model natural language syntax. "
        "This means that productions are built are phrases (XP) which produce "
        "specifiers (YP) and bar-level projections (XBar); these bar-level "
        "projections in turn produce heads (X) and complements (ZP). Certain "
        "lexical productions in the grammar produce words which begin with a "
        "null symbol '\\u2205'; these words are phonetically null and do not "
        "appear in the surface forms of either the input or output sentences, "
        "though they may be important for the syntactic structure of the "
        "sentence. Do not include these null words in your output sentence, "
        "though you may need to reason about them to get the correct "
        f"structure.{agreement_note}\n\n"
        "    You may use any reasoning strategy you like to solve this task, "
        "including identifying the categories of the words in the input "
        "sentence, using the grammar to build a parse tree for the input, and "
        "then following that derivation using the other language's expansions "
        "to produce the output sentence. Feel free to write down intermediate "
        "steps in your reasoning.\n\n"
        "    You will be evaluated based on the string accuracy of the output "
        "sentence, which you should format like the following: `Final answer: "
        "<output sentence>`. If you do not end your response with this "
        "format, you will be marked as incorrect.\n\n"
        "    Here is the synchronous context-free grammar:\n"
        "    ```\n"
        f"    {grammar_str}\n"
        "    ```\n\n"
        f"{examples_note}"
        f"    Here is the input sentence: `{sample}`.\n\n"
        "    Remember to end your response with the format `Final answer: "
        "<output sentence>`."
    )

    return prompt
