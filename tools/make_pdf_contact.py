from pathlib import Path
from PIL import Image, ImageDraw

pages = sorted(Path("tmp/pdf_pages").glob("page-*.jpg"))
thumb = (198, 280)
out = Path("tmp/pdf_contact")
out.mkdir(parents=True, exist_ok=True)
for start in range(0, len(pages), 20):
    images = []
    for path in pages[start:start + 20]:
        image = Image.open(path)
        image.thumbnail(thumb)
        images.append((path, image.copy()))
    sheet = Image.new("RGB", (thumb[0] * 5, thumb[1] * 4), (230, 230, 230))
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(images):
        x, y = (index % 5) * thumb[0], (index // 5) * thumb[1]
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + 4), path.stem, fill=(200, 0, 0))
    sheet.save(out / f"contact_{start + 1:03d}.jpg")
print(len(pages), len(list(out.glob("*.jpg"))))
