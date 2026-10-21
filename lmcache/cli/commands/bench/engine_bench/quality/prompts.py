# SPDX-License-Identifier: Apache-2.0
"""The multi-passage QA prompt used by the answer-quality benchmarks.

Kept in one module so every benchmark that scores answers over the same
passages asks byte-identical questions: a prompt difference between two runs
would read as a quality difference.
"""

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


def compose_qa_messages(
    system_block: str, passage_blocks: list[str], question: str
) -> list[dict[str, str]]:
    """Build the chat messages for one question over its passages.

    Args:
        system_block: :data:`QA_SYSTEM_PROMPT`, chunk-aligned by
            :meth:`ChunkAligner.system_block`.
        passage_blocks: Chunk-aligned passages, in the order to present them.
        question: The question text.

    Returns:
        A single user message:
        ``[system block][passage]…[passage][question template]``.
    """
    content = (
        system_block
        + "".join(passage_blocks)
        + QA_QUESTION_TEMPLATE.format(question=question)
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
