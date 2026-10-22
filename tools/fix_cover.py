"""Move the two heading lines and logo away from the supplied ornamental border."""
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import FloatObject

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'report/assets/cover.pdf'
reader = PdfReader(source)
page = reader.pages[0]
content = page.get_contents()
adjusted = 0
for args, operator in content.operations:
    # Coordinates are in the source's top-down page system, in points.
    if operator == b'cm' and float(args[0]) == .75 and float(args[4]) == 72:
        if any(abs(float(args[5]) - y) < .001 for y in (106.38208, 127.540283)):
            args[5] = FloatObject(float(args[5]) + 30)
            adjusted += 1
    if operator == b're' and abs(float(args[0]) - 246.63782) < .001:
        args[1] = FloatObject(float(args[1]) + 30)
        adjusted += 1
    if operator == b'cm' and abs(float(args[0]) - 101.999992) < .001:
        args[5] = FloatObject(float(args[5]) + 30)
        adjusted += 1
assert adjusted == 4, 'Unexpected cover layout; inspect the source before editing.'
page.replace_contents(content)
writer = PdfWriter()
writer.add_page(page)
with (ROOT / 'report/assets/cover_adjusted.pdf').open('wb') as stream:
    writer.write(stream)
