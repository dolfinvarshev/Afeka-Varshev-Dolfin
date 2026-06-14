# travel-pages

This folder is the **automatic backup target** of the *"Travel HTML Page - AI + Airtable"* n8n automation.

## What's in here

Every time the automation runs, the AI-generated travel guide it produced is committed here as a self-contained HTML file. It happens **before** the page is emailed for approval. This happens on **every activation**, including regenerations, so the folder keeps a full history of all generated versions (approved and rejected alike).

Each page contains, for the place the user typed in Airtable:
- 3 photos matched to the recommended attractions
- 3 attraction recommendations written by the AI
- a static map with a pin on each attraction's exact coordinates

## File naming

```
<AirtableRecordId>-<timestamp>.html
```

Example: `recAb8ZPy4FpBHD68-1781290801017.html`

- The **record id** ties the file back to the row in the Airtable base that triggered it.
- The **timestamp** makes every generation unique - when the user chooses *"Reject and create a new page"*, the new version is saved as a new file instead of overwriting the previous one.

## Viewing a page

The repository is served by GitHub Pages, so every file here is a live web page:

```
https://dolfinvarshev.github.io/Afeka-Varshev-Dolfin/travel-pages/<filename>.html
```

When the user **approves** a page, the link to that specific version is saved in the Airtable base (in the row's `Page URL` field). Files with no link in Airtable are earlier versions that were rejected or replaced.
