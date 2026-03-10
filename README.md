# Journal Processor

End-to-end pipeline for processing ~3 000 double-paged digitised journal scans
of a German ornithologist's field journal.

## Pipeline stages

```
1. Split     double-page scan → left page + right page
2. Preprocess  optional deskew / contrast enhancement
3. Detect    region detection via Gemini 3 Flash Preview
4. Transcribe  per-region text recognition via Gemini 3 Flash Preview
5. Output    Markdown · PAGE XML · ShareGPT JSONL
```

## Output structure

```
output/
├── pages/            # individual page images (left/right)
├── regions/          # per-page region crops + detection JSON
│   ├── scan_001_L/
│   │   ├── r01_ParagraphRegion.png
│   │   └── r02_PageNumberRegion.png
│   └── scan_001_L.json
├── md/               # Markdown page reconstructions
├── pagexml/          # PAGE XML with layout + transcription
├── sharegpt/         # training_data.jsonl
└── summary.json
```

## Region types

| Type | Metadata | Transcription |
|------|----------|---------------|
| ParagraphRegion | line_count | exact line-by-line, `<u>` / `<sup>` markup |
| ListRegion | line_count | exact line-by-line |
| TableRegion | rows, cols | Markdown table |
| ObjectRegion | — | description + object_type |
| PageNumberRegion | page_number | skipped (extracted in detection) |
| MarginaliaRegion | line_count | exact transcription |
| FootnoteRegion | line_count | exact transcription |
| ImageRegion | — | description + drawing_type |

## Quick start

```bash
# Install
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-key"

# Run
python run.py -i /path/to/scans -o /path/to/output

# Options
python run.py -i scans/ -o out/ \
    --workers 8 \
    --max-regions 5 \
    --deskew \
    --enhance-contrast \
    -v
```

## Notes

- The detection prompt instructs Gemini to keep whole journal entries
  (date + city) in a single region where possible.
- Maximum 5 regions per page by default to avoid over-segmentation.
- PageNumberRegion extraction happens during detection, saving one API call.
- ShareGPT output only includes ParagraphRegion, ListRegion, TableRegion,
  and FootnoteRegion.
- Region crops are saved alongside detection JSON for debugging / review.
