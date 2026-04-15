# Page Design Specification — HKDSE Core Math 2012 PDF Processing

## Global (applies to all pages)
### Layout
- Desktop-first, max content width 1200px.
- Primary layout uses CSS Grid for app shell and split panes; Flexbox within components.
- Breakpoints: 1200+ (3-column capable), 768–1199 (2-column), <768 (stacked, panes become tabs).

### Meta Information (defaults)
- Title pattern: `HKDSE Math 2012 – {Page}`
- Description: “Process and study Questions 1–3 with verified, citation-backed answers.”
- Open Graph: title/description match; type `website`.

### Global Styles (design tokens)
- Background: #0B1020 (app) with elevated surfaces #111A33.
- Text: primary #EAF0FF, secondary #A9B4D0.
- Accent: #6EA8FF; Success #3DDC97; Warning #FFC857; Danger #FF5C7A.
- Typography: base 16px; headings 24/20/18; monospace for code/citations.
- Buttons: primary filled accent; secondary outline; hover = +6% brightness; disabled = 40% opacity.
- Links: accent underline on hover.
- Math: KaTeX theme aligned to body font size; ensure line-height 1.4 and overflow-x auto for long formulas.

---

## Page 1 — Workspace
### Layout
- App header (top) + main content (centered) with a two-column grid.
- Left column: sessions and actions; Right column: upload + processing status.

### Meta Information
- Title: “HKDSE Math 2012 – Workspace”
- Description: “Create a session, upload the PDF, and process Questions 1–3.”

### Page Structure
1. Header bar
2. Main grid: (A) Session panel, (B) PDF panel

### Sections & Components
1. Header bar
   - Left: product name.
   - Right: “Current session” indicator (title + last updated).

2. Session panel (left)
   - “New session” button.
   - Session list (cards): title, updated time, status badge (Not processed / Processing / Ready).
   - Card click opens the session’s last-viewed question route.

3. PDF panel (right)
   - Upload dropzone + file picker.
   - Validation messages (wrong file type, upload failed).
   - Processing timeline (stepper): Upload → Extract → Segment Q1–Q3 → Index.
   - “Start processing” button (disabled until PDF present).

4. Q1–Q3 shortcuts (bottom of PDF panel)
   - Three large buttons: “Question 1”, “Question 2”, “Question 3”.
   - Disabled until processing is Ready.

### Responsive behavior
- <768px: session list becomes a top section; PDF panel below; Q buttons become full-width stacked.

---

## Page 2 — Question Viewer
### Layout
- Three-region layout using CSS Grid:
  - Left: question navigation + session info.
  - Center: content split (Extracted view + PDF reference tabs or split pane).
  - Right: chat + verification.

### Meta Information
- Title: “HKDSE Math 2012 – Question {1|2|3}”
- Description: “Read the question, ask citation-backed questions, and verify answers.”

### Page Structure
1. Header bar
2. Body grid: Left rail / Main content / Right rail

### Sections & Components
1. Left rail
   - Question switcher (segmented control): Q1 / Q2 / Q3.
   - Session actions: “Back to workspace”, “Re-run processing” (when needed).

2. Main content
   - View mode toggle: “Extracted” / “PDF” / “Split”.
   - Extracted view
     - Render structured blocks in reading order.
     - Math blocks rendered via KaTeX; provide fallback styling when render fails (show raw LaTeX + error label).
     - Figures: show placeholder with caption (if detected) and a “View in PDF” jump.
   - PDF reference view
     - Embedded PDF renderer.
     - When a citation is clicked, auto-scroll to the cited page and highlight the cited bbox.

3. Right rail
   - Chat panel
     - Message list with roles and timestamps.
     - Each assistant message shows citations (page + snippet) as clickable chips.
     - Input box with “Ask” action; disabled while processing.
   - Verification panel
     - “Verify last answer” button.
     - Result banner: Pass / Needs review.
     - Reasons list and supporting citations.

### Interaction states
- Processing state: skeleton loaders; disable Q switching only if artifacts missing.
- Errors: show non-blocking toast + inline error near affected module.

### Responsive behavior
- 768–1199px: hide left rail into a collapsible drawer.
- <768px: convert three-region layout into tabs: (1) Question, (2) PDF, (3) Chat/Verify.
