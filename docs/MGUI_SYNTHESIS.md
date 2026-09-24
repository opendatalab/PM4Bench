# MGUI composition details

Use the [shared four-task interface](VISION_SYNTHESIS.md):

```bash
pm4bench render-vision --task mgui \
  --dataset-root /path/to/PM4Bench --language en --limit 2 \
  --output-root ./rendered-mgui
```

MGUI reads `data/mgui/<language>.jsonl`, like every other task. Its additional
sources are `metadata/mgui/templates/` (Jinja/JSON) and `metadata/mgui/gt/`.
Planning checks each instruction's question, target, difficulty, title, image
size, bbox, and click coordinates against this metadata, then records hashes
of the template/config/GT inputs. Edited metadata that disagrees with the
manifest fails before rendering. Template text is not resampled.

One screenshot is shared by multiple instructions. Selection and limits apply
to instructions; rendering groups by `(template, language)` and creates each
page once. `images.jsonl` has one row per selected instruction, pointing to a
shared page when appropriate. The generated page GT contains all its original
questions, even if only one was selected. `records` and `images` therefore
need not be equal.

The renderer retains the existing viewport, Jinja escaping, platform-font
stack, browser timing, and DOM target extraction. No explicit `--fonts-root`
is accepted: changing this font policy would be a separate rendering change.
Browser/version information and the font policy are recorded in the report.

The default `--comparison-mode structure` compares selected pages with their
reference GT, including questions, element IDs, and visible text. It ignores
geometry differences caused by platform fonts. Use `--comparison-mode strict`
and optionally `--bbox-tolerance 0.2` for geometric checks. Unselected reference
pages are not reported as missing. A failed comparison preserves the output
and returns a nonzero CLI exit status.

Two historical Jinja templates emit a dictionary-method representation with a
process-specific memory address. Comparison normalizes only that address.
The template behavior is preserved for compatibility with published GT.

For old template-only workflows, `pm4bench render-mgui` and
`pm4bench.mgui.render` forward to the same browser implementation. They are
compatibility APIs, not a second independently maintained renderer.
