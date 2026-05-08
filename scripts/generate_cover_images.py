"""Generate AI cover images for the presentation deck via OpenAI's gpt-image-2.

Uses the same model identifier and Python SDK pattern OpenAI documents in their
cookbook (`client.images.generate(model="gpt-image-2", ...)`). API key is loaded
from .env (gitignored). Outputs PNGs under plots/covers/ which the PPTX builder
embeds as title backgrounds and section dividers.

Generated images are decorative, NOT data plots — gpt-image-2 cannot accurately
render real numbers, so all metric charts come from matplotlib instead. This
script is for the title page, section dividers, and the conclusion cover.

Run: ``python scripts/generate_cover_images.py``  (~$0.50-1.50 in API cost)
"""

import base64
import logging
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COVERS_DIR = PROJECT_ROOT / "plots" / "covers"
COVERS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Map: filename -> (prompt, size, quality)
COVERS: dict[str, tuple[str, str, str]] = {
    "title_cover.png": (
        "Editorial poster, dark navy background with subtle abstract chart-grid texture, "
        "minimalist financial-data aesthetic, four ascending light beams in green, blue, "
        "orange, and gray representing four model arms climbing in stair-steps from left "
        "to right. Clean Bauhaus-inspired geometric composition, no text or words anywhere "
        "in the image, plenty of negative space in the upper third for a title overlay. "
        "Quiet, professional, conference-grade design.",
        "1536x1024",
        "high",
    ),
    "rq1_hero.png": (
        "Abstract editorial illustration: a single bold ascending green curve dramatically "
        "outpacing four flatter neutral-gray reference lines on a deep navy background. "
        "Geometric Bauhaus composition with subtle grid texture, soft cinematic lighting, "
        "gallery-quality minimalism. No text, no numbers, no labels, no axes. Section "
        "divider for a finance research talk slide titled \"Risk-Adjusted Returns.\"",
        "1536x1024",
        "high",
    ),
    "regime_concept.png": (
        "Conceptual abstract illustration of market volatility regimes, three vertical bands "
        "left to right showing calm (cool blue gradient with gentle waves), normal (neutral "
        "gray with mild ripples), and stress (deep crimson with jagged lightning lines). "
        "Minimalist editorial style, painterly textures, no text anywhere, no numbers, no "
        "axis labels, no people, no faces. Used as a section divider for a finance research "
        "talk on portfolio behavior across volatility regimes.",
        "1536x1024",
        "high",
    ),
    "rq3_hero.png": (
        "Atmospheric editorial illustration: a calm midnight horizon over still water "
        "reflecting faint rising charts as light streaks, with distant storm clouds receding "
        "to the right. Conveys tail-risk reduction and downside protection. Cinematic, "
        "moody, minimal, navy and deep blue palette with cool gold highlights. No text, no "
        "numbers, no labels, no people, no faces. Used as a section divider for a finance "
        "research talk slide on downside risk and Conditional VaR.",
        "1536x1024",
        "high",
    ),
    "method_concept.png": (
        "Abstract data-flow illustration: multiple thin colored ribbons (green, blue, "
        "orange, gray) flowing left to right through a stylized funnel and emerging on the "
        "right as a single bright braided cable. Suggests four model arms feeding the same "
        "Markowitz pipeline. Editorial graphic-design aesthetic, deep navy background, soft "
        "lighting, no text anywhere, no numbers, no logos.",
        "1536x1024",
        "medium",
    ),
    "tuning_concept.png": (
        "Conceptual still-life illustration: an analog tuning dial and a stack of nested "
        "calibration rings glowing on a dark workbench, with faint chart traces in the "
        "background suggesting a model being tuned. Editorial, moody, navy and amber palette. "
        "Conveys hyperparameter and architecture tuning without literal computer imagery. "
        "No text, no numbers, no logos, no UI screens.",
        "1536x1024",
        "medium",
    ),
    "complexity_concept.png": (
        "Stylized geometric staircase rising from lower-left to upper-right, four broad "
        "steps in green, blue, orange, and gray, each step slightly higher than the last. "
        "Set against a deep navy background with soft chart-grid texture. Editorial poster "
        "design, Bauhaus-inspired clean shapes, plenty of negative space, cinematic "
        "lighting. No text, no numbers, no labels.",
        "1536x1024",
        "high",
    ),
    "conclusion_cover.png": (
        "Aspirational closing image for a quantitative-finance research talk: a single bold "
        "upward path made of light, rising over a stylized ledger of soft horizontal lines, "
        "set against a near-black background. Four faint companion lines trail behind the "
        "main path in green, blue, orange, and gray, suggesting the four-arm model "
        "comparison. Editorial, minimal, no text, no numbers, plenty of upper-area negative "
        "space for a closing-statement overlay.",
        "1536x1024",
        "high",
    ),
}


def generate_one(client: OpenAI, filename: str, prompt: str, size: str, quality: str) -> Path:
    out_path = COVERS_DIR / filename
    if out_path.exists():
        logger.info("skip %s (already exists)", out_path)
        return out_path

    logger.info("generating %s (size=%s, quality=%s)", filename, size, quality)
    result = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        size=size,
        quality=quality,
    )
    b64 = result.data[0].b64_json
    image = Image.open(BytesIO(base64.b64decode(b64)))
    image.save(out_path, format="PNG")
    logger.info("wrote %s", out_path)
    return out_path


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment. Put it in .env (gitignored)."
        )
    client = OpenAI(api_key=api_key)

    for filename, (prompt, size, quality) in COVERS.items():
        try:
            generate_one(client, filename, prompt, size, quality)
        except Exception as exc:  # broad catch: API errors, rate limits, etc.
            logger.error("failed to generate %s: %s", filename, exc)
            raise

    logger.info("All covers in %s", COVERS_DIR.resolve())


if __name__ == "__main__":
    main()
