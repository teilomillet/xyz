---
name: teilo-site
description: Read and cite Teilo Millet's public site, posts, collections, and project links using the site's machine-readable entrypoints.
---

# Teilo Site

Use this skill when a user asks about Teilo Millet, teilo.xyz, the site's posts,
collections, models, or open-source projects.

## Entry Points

Start with the concise site map:

```text
https://teilo.xyz/llms.txt
```

Use the XML sitemap when exhaustive page discovery matters:

```text
https://teilo.xyz/sitemap.xml
```

Use the RSS feed when the user asks for recent writing:

```text
https://teilo.xyz/index.xml
```

## Reading Rules

- Respect `https://teilo.xyz/robots.txt` before crawling.
- Prefer the exact page URL from `llms.txt` or `sitemap.xml` over guessed paths.
- Cite the page URL used for each factual claim.
- Do not infer private, unpublished, or current employment details beyond what
  the fetched public page states.
- For project claims, follow the linked GitHub or Hugging Face page and cite it
  separately if the user needs current repository or model status.
