# SPDX-License-Identifier: Apache-2.0
"""The multi-passage QA prompt used by the answer-quality benchmarks.

Kept in one module so every benchmark that scores answers over the same
passages asks byte-identical questions: a prompt difference between two runs
would read as a quality difference.

Two versions exist. :data:`ANSWER_TAG_PROMPT` (``bench quality``) asks for
``<answer>`` tags. :data:`QA_SYSTEM_PROMPT` / :data:`QA_QUESTION_TEMPLATE`
(``rag-qa-quality``) ask for ``<final_answer>`` tags, which gpt-oss trips
over: its response format names the answer channel ``final``, and a prompt
full of ``final_answer`` leads it to write ``<|channel|>final_answer>…``,
dropping the opening tag. ``extract_final_answer`` accepts both tags and that
form.
"""

# Standard
from dataclasses import dataclass

QA_SYSTEM_PROMPT = (
    "Answer the question using only the given passages. You may reason "
    "through the problem, but put only the concise final answer between "
    "<final_answer> and </final_answer>. Always emit both tags.\n\n"
    "The following are the given passages.\n"
)

QA_QUESTION_TEMPLATE = (
    "\n\nAnswer the question using only the passages above. End with exactly "
    "one concise answer in this form: <final_answer>answer</final_answer>.\n\n"
    "Question: {question}\nAnswer:"
)


@dataclass(frozen=True)
class QAPrompt:
    """The instructions around a question's passages.

    Attributes:
        system: Placed before the passages (then chunk-aligned).
        question_template: Placed after them; ``{question}`` is filled in.
    """

    system: str
    question_template: str


ANSWER_TAG_PROMPT = QAPrompt(
    system=(
        "Answer the question using only the given passages. You may reason "
        "through the problem, but put only the concise answer between "
        "<answer> and </answer>. Always emit both tags.\n\n"
        "The following are the given passages.\n"
    ),
    question_template=(
        "\n\nAnswer the question using only the passages above. End with "
        "exactly one concise answer in this form: <answer>answer</answer>.\n\n"
        "Question: {question}\nAnswer:"
    ),
)
"""The prompt with ``<answer>`` tags, which no checked chat format uses."""


def compose_qa_messages(
    system_block: str,
    passage_blocks: list[str],
    question: str,
    question_template: str = QA_QUESTION_TEMPLATE,
) -> list[dict[str, str]]:
    """Build the chat messages for one question over its passages.

    Args:
        system_block: A :class:`QAPrompt` system text, chunk-aligned by
            :meth:`ChunkAligner.system_block`.
        passage_blocks: Chunk-aligned passages, in the order to present them.
        question: The question text.
        question_template: The text after the passages, with a
            ``{question}`` field; the system block's prompt version.

    Returns:
        A single user message:
        ``[system block][passage]…[passage][question template]``.
    """
    content = (
        system_block
        + "".join(passage_blocks)
        + question_template.format(question=question)
    )
    return [{"role": "user", "content": content}]


def compose_store_messages(
    system_block: str, passage_blocks: list[str]
) -> list[dict[str, str]]:
    """Build the chat messages that store passages in the cache.

    The stored prompt carries the same system block as the measured one, so
    the passages sit at the same chunk phase in both.

    Args:
        system_block: The chunk-aligned system block.
        passage_blocks: Chunk-aligned passages to store, in order.

    Returns:
        A single user message: ``[system block][passage]…[passage]``.
    """
    return [{"role": "user", "content": system_block + "".join(passage_blocks)}]
