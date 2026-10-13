# Landing page

`index.html` is the project page for R2-D2 Redux, served at
<https://jayinaksha.github.io/agentic-bot/> from `main` / `/docs`. The empty `.nojekyll`
file tells Pages to serve the directory as-is.

It is built on **Plate**, the design system shared by every site under
`jayinaksha.github.io` — this page is where Plate's look came from. The source of truth
for Plate lives in the portfolio repo (`design-system/`); `assets/plate/` here is a
vendored copy, so a change there never breaks this page by surprise.

| Path | What it holds |
| --- | --- |
| `assets/plate/` | `plate.css`, `plate.js`, and the OFL webfonts |
| `assets/img/` | The annotated hero photo, the semantic map, and the build photos |
| `assets/media/` | Short muted test-run clips, each with a poster frame |
| `assets/og.jpg`, `assets/favicon.svg` | Share card and icon |

## Editing it

- **Accent and palette** — Plate tokens. Override `--accent` on `:root` to recolour the page.
- **Copy** — each `<section>` has an id matching the top-bar links.
- **Architecture diagram** — the inline `<svg>` in `#path`, drawn on a 1120 × 500 grid.
  Node boxes are `<rect class="nb">`; the animated dots follow the `<path>` elements by id
  (`#w1`…`#w9`) and are hidden under `prefers-reduced-motion`.
- **Long-form detail** — the hard parts, the service sheet and the run order sit inside
  `<details class="more">` blocks in `#more`, so the page reads in a minute but nothing is lost.

Check both themes after any colour change; the page follows the reader's system setting
until they use the toggle.
