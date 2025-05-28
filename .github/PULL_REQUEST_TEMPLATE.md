FILL IN THE PR DESCRIPTION HERE

FIX #xxxx (*link existing issues this PR will resolve*)

**BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE**

---

<details>
<!-- inside this <details> section, markdown rendering does not work, so we use raw html here. -->
<summary><b> PR Checklist (Click to Expand) </b></summary>

<p>Thank you for your contribution to LMCache! Before submitting the pull request, please ensure the PR meets the following criteria. This helps us maintain the code quality and improve the efficiency of the review process.</p>

<h3>PR Title and Classification</h3>
<p>Please try to classify PRs for easy understanding of the type of changes. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:</p>
<ul>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[Doc]</code> for documentation fixes and improvements.</li>
    <li><code>[Model]</code> for adding a new model or improving an existing model. Model name should appear in the title.</li>
    <li><code>[Core]</code> for changes in the core LMCache logic (e.g., <code>LMCacheEngine</code>, <code>Backend</code> etc.)</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If the PR spans more than one category, please include all relevant prefixes.</p>

<h3>Code Quality</h3>

<p>The PR need to meet the following code quality standards:</p>

<ul>
    <li>The code need to be well-documented to ensure future contributors can easily understand the code.</li>
    <li> Please include sufficient tests to ensure the change is stay correct and robust. This includes both unit tests and integration tests.</li>
</ul>

<h3>What to Expect for the Reviews</h3>

To create a new tag for lmcache (Note: `v` prefix is required):
`git tag vx.x.x`
`git push origin vx.x.x` (same version again)

For example:
`git tag v0.3.0`
`git push origin v0.3.0`

In case the workflow fails, delete the tag and try again:
`git tag -d vx.x.x`
`git push origin :refs/tags/vx.x.x`

For example:
`git tag -d v0.3.0`
`git push origin :refs/tags/v0.3.0`

To create a new release and publish `lmcache` Python package to PyPi:
`git remote add upstream git@github.com:LMCache/LMCache.git`
`gh release create vx.x.x --repo LMCache/LMCache --title "vx.x.x" --notes "<Add description>"`

For example:
`git remote add upstream git@github.com:LMCache/LMCache.git`
`gh release create v0.3.0 --repo LMCache/LMCache --title "v0.3.0" --notes "LMCache v0.3.0 is a feature release. Users are encouraged to upgrade for the best experience."`

> [!TIP]
> The creation of a release and subsequent tag generation can be done alternatively from the LMCache [releases](https://github.com/LMCache/LMCache/releases) page.

We aim to address all PRs in a timely manner. If no one reviews your PR within 5 days, please @-mention one of KuntaiDu, ApostaC or YaoJiayi.
