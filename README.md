<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="lmcache logo">
  </p>
  
  [![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://docs.lmcache.ai/)
  [![PyPI](https://img.shields.io/pypi/v/lmcache)](https://pypi.org/project/lmcache/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmcache)](https://pypi.org/project/lmcache/)
  [![Unit Tests](https://badge.buildkite.com/ce25f1819a274b7966273bfa54f0e02f092c3de0d7563c5c9d.svg)](https://buildkite.com/lmcache/lmcache-unittests)
  [![Code Quality](https://github.com/lmcache/lmcache/actions/workflows/code_quality_checks.yml/badge.svg?branch=dev&label=tests)](https://github.com/LMCache/LMCache/actions/workflows/code_quality_checks.yml)
  [![Integration Tests](https://badge.buildkite.com/108ddd4ab482a2480999dec8c62a640a3315ed4e6c4e86798e.svg)](https://buildkite.com/lmcache/lmcache-vllm-integration-tests)

   <br />

  [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10841/badge)](https://www.bestpractices.dev/projects/10841)
  [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/LMCache/LMCache/badge)](https://scorecard.dev/viewer/?uri=github.com/LMCache/LMCache)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache/)
  [![GitHub commit activity](https://img.shields.io/github/commit-activity/w/LMCache/LMCache)](https://github.com/LMCache/LMCache/graphs/commit-activity)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/lmcache)](https://pypi.org/project/lmcache/)
  [![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UC58zMz55n70rtf1Ak2PULJA)](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

</div>


--------------------------------------------------------------------------------

| [**Blog**](https://blog.lmcache.ai/)
| [**Documentation**](https://docs.lmcache.ai/)
| [**Join Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
| [**Interest Form**](https://forms.gle/MHwLiYDU6kcW3dLj7)
| [**Roadmap**](https://github.com/LMCache/LMCache/issues/1253)

🔥 **NEW: For enterprise-scale deployment of LMCache and vLLM, please check out vLLM [Production Stack](https://github.com/vllm-project/production-stack). LMCache is also officially supported in [llm-d](https://github.com/llm-d/llm-d/) and [KServe](https://github.com/kserve/kserve)!**


## Summary

LMCache is an **LLM** serving engine extension to **reduce TTFT** and **increase throughput**, especially under long-context scenarios. By storing the KV caches of reusable texts across various locations, including (GPU, CPU DRAM, Local Disk), LMCache reuses the KV caches of **_any_** reused text (not necessarily prefix) in **_any_** serving engine instance. Thus, LMCache saves precious GPU cycles and reduces user response delay.  

By combining LMCache with vLLM, developers achieve 3-10x delay savings and GPU cycle reduction in many LLM use cases, including multi-round QA and RAG.

![performance](https://github.com/user-attachments/assets/86137f17-f216-41a0-96a7-e537764f7a4c)

## Features

- [x] 🔥 Integration with vLLM v1 with the following features:
  * High performance CPU KVCache offloading
  * Disaggregated prefill
  * P2P KVCache sharing
- [x] LMCache is supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve) 
- [x] Stable support for non-prefix KV caches
- [x] Storage support as follows:
  * CPU
  * Disk
  * [NIXL](https://github.com/ai-dynamo/nixl)
- [x] Installation support through pip and latest vLLM

## Installation

To use LMCache, simply install `lmcache` from your package manager, e.g. pip:

```bash
pip install lmcache
```

Works on Linux NVIDIA GPU platform.

More [detailed installation instructions](https://docs.lmcache.ai/getting_started/installation) are available in the docs, particularly if you are not using the latest stable version of vllm or using another serving engine with different dependencies. Any "undefined symbol" or torch mismatch versions can be resolved in the documentation. 

## Getting started

The best way to get started is to checkout the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the docs.

## Documentation

Check out the LMCache [documentation](https://docs.lmcache.ai/) which is available online.

We also post regularly in [LMCache blogs](https://blog.lmcache.ai/).

## Examples

Go hands-on with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples),
demonstrating how to address different use cases with LMCache.

## Interested in Connecting?

Fill out the [interest form](https://forms.gle/mQfQDUXbKfp2St1z7), [sign up for our newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter), [join LMCache slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ), [check out LMCache website](https://lmcache.ai/), or [drop an email](mailto:contact@lmcache.ai), and our team will reach out to you!

## Community meeting

The [community meeting]( https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09) for LMCache is hosted bi-weekly. All are welcome to join!

Meetings are held bi-weekly on: Tuesdays at 9:00 AM PT – [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)

We keep notes from each meeting on this [document](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow) for summaries of standups, discussion, and action items.

Recordings of meetings are available on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome and value all contributions and collaborations.  Please check out [Contributing Guide](CONTRIBUTING.md) on how to contribute.

We continually update [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627)

## Citation

If you use LMCache for your research, please cite our papers:

```
@inproceedings{liu2024cachegen,
  title={Cachegen: Kv cache compression and streaming for fast large language model serving},
  author={Liu, Yuhan and Li, Hanchen and Cheng, Yihua and Ray, Siddhant and Huang, Yuyang and Zhang, Qizheng and Du, Kuntai and Yao, Jiayi and Lu, Shan and Ananthanarayanan, Ganesh and others},
  booktitle={Proceedings of the ACM SIGCOMM 2024 Conference},
  pages={38--56},
  year={2024}
}

@article{cheng2024large,
  title={Do Large Language Models Need a Content Delivery Network?},
  author={Cheng, Yihua and Du, Kuntai and Yao, Jiayi and Jiang, Junchen},
  journal={arXiv preprint arXiv:2409.13761},
  year={2024}
}

@inproceedings{10.1145/3689031.3696098,
  author = {Yao, Jiayi and Li, Hanchen and Liu, Yuhan and Ray, Siddhant and Cheng, Yihua and Zhang, Qizheng and Du, Kuntai and Lu, Shan and Jiang, Junchen},
  title = {CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion},
  year = {2025},
  url = {https://doi.org/10.1145/3689031.3696098},
  doi = {10.1145/3689031.3696098},
  booktitle = {Proceedings of the Twentieth European Conference on Computer Systems},
  pages = {94–109},
}
```

## Socials

[Linkedin](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true) | [Twitter](https://x.com/lmcache) | [Youtube](https://www.youtube.com/@LMCacheTeam)

## License

The LMCache codebase is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for details.
