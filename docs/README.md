# Landing page

`index.html` is the project landing page for R2-D2 Redux — a single, self-contained
file. Fonts (IBM Plex Sans/Mono, Saira Condensed — all SIL OFL) and every image are
embedded as data URIs, so it has zero external requests and works offline, from a
USB stick, or as an email attachment.

## Viewing it

Open `docs/index.html` in any browser.

## Publishing it

The page is static, so anything that serves files will host it. To use GitHub Pages:
**Settings → Pages → Source: Deploy from a branch → `main` / `/docs`**. It will be
served at `https://jayinaksha.github.io/agentic-bot/`.

## Editing it

All the markup and CSS sit at the top of the file and are ordinary HTML — the long
base64 strings are confined to the five `@font-face` rules and the `<img src>`
attributes, and nothing else refers to them.

- **Colours** — the three `:root` blocks at the top. The first is the dark palette,
  the other two are the light palette (one for the OS preference, one for an explicit
  toggle). Change a colour in all the blocks it appears in.
- **Copy** — each `<section>` is labelled with a comment (`══ 05 TRACE ══`).
- **Architecture diagram** — the inline `<svg>` in `§ 02`, drawn on a 1120 × 500 grid.
  Node boxes are `<rect class="nb">` with `<text>` beside them; the animated dots
  follow the `<path>` elements by id (`#w1`…`#w9`).

The page renders in whichever theme the reader's browser is set to, so check both
after any colour change.
