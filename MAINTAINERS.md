# Maintainers

The following is the current list of project maintainers.

- @apostacheng@gmail.com, Yihua Cheng, UChicago & Tensormesh, Committer
- @jiayi3@uchicago.edu, Jiayi Yao, UChicago, Committer
- @kuntai@uchicago.edu, Kuntai Du, UChicago & Tensormesh, Committer
- @martin.hickey@ie.ibm.com, Martin Hickey, IBM, Committer
- @HUAIZHEN001@e.ntu.edu.sg, Hunter Zhang, Bytedance, Committer
- @307499405@qq.com, Baolong Mao, Tensormesh, Committer
- @1179548172@qq.com, Chunxiao Zheng, Tensormesh, Committer
- @shaotingf@uchicago.edu, Shaoting Feng, UChicago, Committer
- @slshen@uchicago.edu, Samuel Shen, UChicago & Tensormesh, Committer
- @dongjoo.seo1@samsung.com, Dongjoo Seo, Samsung, Committer
- @andy.luo@amd.com, Andy Luo, AMD, Committer
- @tony.lin@intel.com, Tony Lin, Intel, Committer
- @crclq2018@gmail.com, Riggins Chen, Nvida, Committer
- @rzhan229@ucsc.edu, Rui Zhang, UCSC & Tensormesh, Committer
- @deng451e@gmail.com, Weishu Deng, Tensormesh, Committer
- @wenxin.zhang@intel.com, Wenxin Zhang, Intel, Committer
- @ayw.sirius19@gmail.com, Yuwei An, RadixArk, Committer
- @hezhengfei1999@gmail.com, Zhengfei He, Mujin.Inc, Committer

## Code area reviewers

Reviewer ownership is documented in `MAINTAINERS.md` files next to the code it
covers. These files help contributors find a knowledgeable reviewer without
automatically subscribing everyone to every pull request.

- [LMCache Python package](lmcache/MAINTAINERS.md)
- [Native extensions](csrc/MAINTAINERS.md)
- [Tests](tests/MAINTAINERS.md)
- [Documentation](docs/MAINTAINERS.md)
- [Examples](examples/MAINTAINERS.md)
- [GitHub workflows and automation](.github/MAINTAINERS.md)
- [Buildkite](.buildkite/MAINTAINERS.md)
- [Kubernetes operator](operator/MAINTAINERS.md)
- [Rust components](rust/MAINTAINERS.md)

When several entries match a change, start with the most specific path. These
lists are routing guidance, not a requirement that every listed person approve
the change.

### Repository-level and cross-module paths

| Path | Reviewers |
| --- | --- |
| `pyproject.toml`, `setup.py` | @sammshen, @ApostaC, @deng451e, @hickeyma |
| `setup_extensions/build_profiles/rocm.py` | @andyluo7 |
| `docker/Dockerfile.rocm`, `docker/Dockerfile.rocm-lightweight` | @andyluo7 |
