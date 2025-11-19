import os

CHINESE_TOKENIZER_BACKEND = os.getenv("TROM_CHINESE_BACKEND", "pkuseg").lower()
JAPANESE_TOKENIZER_BACKEND = os.getenv("TROM_JAPANESE_BACKEND", "sudachipy").lower()

TOKENIZER_CACHE_SIZE = int(os.getenv("TROM_TOKENIZER_CACHE_SIZE", 2048))
ROMANIZER_CACHE_SIZE = int(os.getenv("TROM_CACHE_SIZE", 10000000))

NO_ICU = bool(int(os.getenv("TROM_NO_ICU", "0")))