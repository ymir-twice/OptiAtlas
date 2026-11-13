# OptiAtlas project website

The standalone project webpage is in `site/`. It is a real static website with
the author-provided demo video, visual comparisons, method description, and
benchmark results. It needs no Node.js, build service, analytics, or API key.

Its intended GitHub Pages address is:

**https://ymir-twice.github.io/OptiAtlas/**

This address becomes live only after GitHub Pages is enabled and a deployment
succeeds. Committing the files, or viewing the README, does not publish it.

## Preview without publishing

From the repository root:

```bash
python3 scripts/package_project_site.py --check
python3 -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8000/` on that machine. The page also works by opening
`site/index.html` directly, provided all sibling assets are retained.

To create a transferable, self-contained copy:

```bash
python3 scripts/package_project_site.py --output /tmp/optiatlas-project-site.zip
```

Extract the ZIP and open its `index.html`, or serve that extracted directory.
The ZIP contains only webpage assets, not the repository's training code, private
data, checkpoints, or credentials. The original demo video remains unmodified.

## Enable GitHub Pages when ready to publish

GitHub Pages is normally public even when the source repository is private.
The private repository and Hugging Face artifacts can remain private; files
inside `site/`, including the video and selected images, become publicly visible.
Publishing from a private repository under a personal account requires GitHub
Pro or another supported paid plan. A free personal account must use a separate
public website repository or defer publication; do not make the research-code
repository public merely to enable Pages.

These are GitHub's documented access and plan rules:
[publishing sources and visibility](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

For a supported plan, the repository owner performs this one-time setup:

1. Open [Settings → Pages](https://github.com/ymir-twice/OptiAtlas/settings/pages).
2. Under **Build and deployment → Source**, select **GitHub Actions**.
3. Open **Actions → Publish OptiAtlas project website → Run workflow**.
4. Select branch **main** and check **Publish the site/ project webpage for
   public internet access** only when that audience is intended.
5. Wait for the deployment to succeed. Its `github-pages` environment provides
   the actual published URL.

The workflow is intentionally manual and defaults its publication confirmation
to false. Ordinary pushes never trigger publication. Re-run the workflow after
later webpage changes. Only `site/` is uploaded as the Pages artifact; no deploy
key or personal access token is required by the workflow itself. The existing
SSH deploy key can push code but cannot change repository Pages settings.

The workflow follows GitHub's
[custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
If private repositories cannot use Pages on the current plan, GitHub will show
that limitation in Settings. Do not change visibility or purchase a plan without
the owner's decision.

## Scope and paths

The page uses relative asset paths so it works under the `/OptiAtlas/` project
prefix as well as a local preview. Keep `index.html`, styles, scripts, JSON, and
`assets/` together. The published resource links may require permission while
the code, datasets, and weights are private. A link to a private artifact does
not expose that artifact.

The benchmark table identifies the selected run separately from repeated-run
statistics. The final 80.54% model and strict Dense/Agent controls must remain
distinct when updating the site.
