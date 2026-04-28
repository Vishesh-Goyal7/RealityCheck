import argparse, json
from pathlib import Path
from claim_extractor import extract_claims

ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
with open(args.input,"r",encoding="utf-8") as f: records=json.load(f)
out=extract_claims(records)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
with open(args.output,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
print(f"Saved {len(out)} records to {args.output}")
