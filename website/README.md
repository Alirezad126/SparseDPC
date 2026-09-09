# Le Roi du Smoked Meat — modern website

A modern, responsive, bilingual (FR/EN) redesign of leroidusmokedmeat.com.
Static site: no build step, no dependencies. Open `index.html` or serve the folder.

```
python3 -m http.server -d website 8000
```

## Files

- `index.html` — single-page site: hero, story, menu categories, gallery, visit, contact, footer
- `styles.css` — design system (custom properties), layout, responsive rules
- `app.js` — mobile nav, FR/EN toggle, footer year

## Language toggle

Every translatable element carries `data-fr` and `data-en` attributes; the FR/EN
button swaps `textContent` and persists the choice in `localStorage`. To add copy,
add both attributes — no JS changes needed.

## Content status — please review

The original site could not be crawled from the build environment (the domain is
blocked by the network egress proxy), so the copy was written from publicly
verifiable facts only:

- Name, opened 1954, 6705 rue Saint-Hubert, Montréal (Rosemont–La Petite-Patrie /
  Plaza Saint-Hubert), phone 514 273-7566, roughly 10:00–24:00 daily
- Menu *categories* only: smoked meat, poutine, pizza, souvlaki/BBQ chicken,
  spaghetti, clubs & subs; dine-in, takeout, free neighbourhood delivery

**No individual dish names or prices were invented.** Before going live:

1. Replace the menu category blurbs in `#menu-section` with the real menu and prices.
2. Confirm the opening hours in `#visiter` (the hero and footer reference them too).
3. Replace the CSS gradient placeholders (`.figure`) with real photography —
   swap the `background` for an `<img>` or `background-image`.
4. Wire the "Commander" buttons to the real online-ordering flow if one exists;
   they currently link to the phone number.
