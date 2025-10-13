import os
import sys
from pathlib import Path
from timeit import default_timer as timer

from datasets import Dataset

import trom

if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/000_00000.parquet"
    parquet_path = str(Path(parquet_path).expanduser().absolute())
    
    print("Loading dataset...")
    NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", -1))
    ds = Dataset.from_parquet("/Users/lennartkeller/Downloads/000_00000.parquet")
    if NUM_SAMPLES > 0:
        ds = ds.select(range(NUM_SAMPLES))
    print(f"Using {len(ds)} samples")

    NUM_PROC = os.getenv("NUM_PROC", os.cpu_count())
    print("Using num_proc =", NUM_PROC)



    uromanizer_py = trom.URomanizer()
    uromanizer_rust = trom.URomanizer(backend="uroman_rs")

    print("Profiling Python backend...")
    start = timer()
    ds = ds.map(lambda x: {"text_py": uromanizer_py(x["text"])}, num_proc=NUM_PROC, load_from_cache_file=False)
    end = timer()
    print(f"Python backend took {end - start} seconds")

    print("Profiling Rust backend...")
    start = timer()
    ds = ds.map(lambda x: {"text_rs": uromanizer_rust(x["text"])}, num_proc=NUM_PROC, load_from_cache_file=False)
    end = timer()
    print(f"Rust backend took {end - start} seconds")

    def find_mismatches(example):
        return example["text_py"] != example["text_rs"]
    
    ds = ds.filter(find_mismatches, num_proc=NUM_PROC)
    num_mismatches = len(ds)
    if num_mismatches:
        print(f"Found {num_mismatches} mismatches saving to 'mismatches.parquet'")
        ds.to_parquet("missmatches.parquet")
    else:
        print("No mismatches found!")