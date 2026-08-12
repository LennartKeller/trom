# trom

**trom** (text romanization) is a unified front-end for romanizing text in (almost) any language or script system. It wraps several battle-tested romanization backends behind one consistent API and picks sensible per-language defaults for tokenization and segmentation.

## Backends

| Backend | Implementation | Notes |
|---|---|---|
| `uroman` | [uroman](https://github.com/isi-nlp/uroman) (Python) or [uroman-rs](https://github.com/LennartKeller/uroman-rs) (Rust) | Universal romanizer, covers virtually all scripts (e.g. Cherokee) |
| `uconv` / ICU | In-process [PyICU](https://pypi.org/project/PyICU/) `Any-Latin` transliteration, falling back to the `uconv` CLI | Fast, high-quality output for the scripts ICU supports |
| `unified` | ICU first, uroman as fallback | Tries ICU; if non-Latin letters remain, retries with uroman |

## Installation

```bash
uv add "trom @ git+https://github.com/LennartKeller/trom.git"
```

### Reproducibility

For the ICU backend you need PyICU (already a dependency) or the `uconv` binary (`brew install icu4c` / `apt install icu-devtools`). On macOS, see `install_icupy_macos.sh` for building PyICU against Homebrew's ICU.

## Quickstart

```python
import trom

# Unified romanizer: ICU first, uroman fallback
romanizer = trom.UnifiedRomanizer()
romanizer("Привет мир")            # 'Privet mir'
romanizer("你好世界", lcode="cmn")  # 'nǐ hǎo shì jiè'

# Pure uroman (Python or Rust backend)
uromanizer = trom.URomanizer(backend="rust")
uromanizer("Γεια σου κόσμε")

# Plain cached functions
trom.uromanize("मुझे", lcode="hin")
trom.uconv_romanize("こんにちは")
```

### Language-aware romanizers

`get_romanizer_for_lang` configures a romanizer with per-language defaults (e.g. character-level segmentation for Chinese/Japanese, language-specific tokenizers for pre-processing):

```python
from trom import get_romanizer_for_lang

romanizer = get_romanizer_for_lang(
    "cmn",
    romanizer_cls="unified",       # 'uroman' | 'uconv' | 'unified'
    preprocessor="tokenize",       # None | 'tokenize' | 'aggressive' | callable
    postprocessor="aggressive",    # None | 'aggressive' | callable
)
romanizer("你好，世界！", lcode="cmn")
```

Language-specific word tokenizers are used where whitespace tokenization fails: jieba/pkuseg/thulac (Chinese), SudachiPy or spaCy (Japanese), spaCy (Korean), pycantonese (Cantonese), pythainlp (Thai), laonlp (Lao), khmer-nltk (Khmer), and Moses elsewhere.

## CLI

```bash
# Direct arguments
trom -l rus "Привет мир"

# Choose romanizer and backend
trom -l cmn -r uroman -b rust "你好世界"

# Files / pipes (line by line)
cat corpus.txt | trom -l ell > corpus.rom.txt
trom -l jpn -i input.txt -o output.txt
```

See `trom --help` for all options.

## Configuration

Environment variables (read at import time):

| Variable | Default | Effect |
|---|---|---|
| `TROM_CHINESE_BACKEND` | `pkuseg` | Chinese tokenizer: `pkuseg`, `jieba`, or `thulac` |
| `TROM_JAPANESE_BACKEND` | `sudachipy` | Japanese tokenizer: `sudachipy` or spaCy |
| `TROM_CACHE_SIZE` | `10000000` | `lru_cache` size for romanization calls |
| `TROM_TOKENIZER_CACHE_SIZE` | `2048` | `lru_cache` size for tokenizer calls |
| `TROM_NO_ICU` | `0` | Set to `1` to disable in-process PyICU and force the `uconv` CLI |

## Development

```bash
uv sync
uv run pytest tests/   # full suite (~3 min, loads tokenizer models)
./format_code.sh       # ruff format + import sorting
```
