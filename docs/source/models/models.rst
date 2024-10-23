.. _models:

Supported Models
=============================

Currently, LMCache supports the following models:

* `Qwen/Qwen-7B <https://huggingface.co/Qwen/Qwen-7B>`_
* `THUDM/glm-4-9b-chat <https://huggingface.co/THUDM/glm-4-9b-chat>`_
* `lmsys/longchat-7b-v1.5-32k <https://huggingface.co/lmsys/longchat-7b-v1.5-32k>`_
* `meta-llama/Llama-3.1-8B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_
* `mistralai/Mistral-7B-Instruct-v0.2 <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`_

To use vLLM's offline inference with LMCache, for any model, use the required model 
card name as on Huggingface.

.. code-block:: python

    import lmcache_vllm.vllm as vllm
    from lmcache_vllm.vllm import LLM 

    # model card (Huggingface model card format name)
    model_card = "insert here"

    # Load the model
    model = LLM.from_pretrained(model_card)

    # Use the model
    model.generate("Hello, my name is", max_length=100)

.. note:: 
    To use the models, you might often require setting up a Huggingface-login token, after 
    you accept the terms and conditions of the model. To do so, you can add the following
    to the top of your Python script:

.. code-block:: python

    from huggingface_hub import login
    login()

    # You will now be prompted to enter your Huggingface login credentials.

For more information on Huggingface login, please refer to the `Huggingface documentation <https://huggingface.co/docs/huggingface_hub/en/quick-start>`_.