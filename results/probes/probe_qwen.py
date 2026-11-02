"""CPU probe of Qwen3-VL-2B-Instruct on real NIH radiographs: token budget, first-token coverage, answers."""
import json, sys, time, math
from pathlib import Path
import torch, pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

repo, img_dir, meta_csv, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
log = open(out_path, "w", encoding="utf-8")
def emit(**kw):
    log.write(json.dumps(kw) + "\n"); log.flush(); print(json.dumps(kw)[:300], flush=True)

prompts = json.load(open(repo / "cxrdev" / "prompts.json", encoding="utf-8"))
dev = prompts["devices"]["ett"]
P = {k: v["template"].replace("{device}", dev) for k, v in prompts["prompts"].items() if k in ("P0", "P1", "P2hi")}
model_id = "Qwen/Qwen3-VL-2B-Instruct"
torch.set_num_threads(8)
proc = AutoProcessor.from_pretrained(model_id)
ip = proc.image_processor
emit(stage="processor", cls=type(ip).__name__, size=str(getattr(ip, "size", None)), patch=str(getattr(ip, "patch_size", None)), merge=str(getattr(ip, "merge_size", None)))
ip.size = {"shortest_edge": 1024 * 1024, "longest_edge": 1024 * 1024}
meta = pd.read_csv(meta_csv).set_index("Image Index")
avail = sorted(p.name for p in img_dir.glob("*.png"))
ap = [n for n in avail if meta.loc[n, "View Position"] == "AP"]
pa = [n for n in avail if meta.loc[n, "View Position"] == "PA"]
picks = [ap[len(ap) // 2], pa[len(pa) // 3]]
tok = proc.tokenizer
def ids(forms):
    out = []
    for f in forms:
        e = tok.encode(f, add_special_tokens=False)
        if len(e) == 1 and e[0] not in out: out.append(e[0])
    return out
yes_ids = ids(["Yes", "yes", " Yes", " yes", "YES", " YES"]); no_ids = ids(["No", "no", " No", " no", "NO", " NO"])
emit(stage="ids", yes=yes_ids, no=no_ids, yes_tok=[tok.decode([i]) for i in yes_ids], no_tok=[tok.decode([i]) for i in no_ids])
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)
model.eval()
emit(stage="loaded", seconds=round(time.time() - t0, 1), dtype=str(model.dtype))
image_token_id = getattr(model.config, "image_token_id", None)

def run(img, prompt_id):
    content = ([{"type": "image", "image": img}] if img is not None else []) + [{"type": "text", "text": P[prompt_id]}]
    msgs = [{"role": "user", "content": content}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    grid = inputs.get("image_grid_thw")
    n_img_tok = int((inputs["input_ids"] == image_token_id).sum()) if image_token_id is not None else -1
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                             output_logits=True, return_dict_in_generate=True)
    lp = torch.log_softmax(out.logits[0][0].float(), dim=-1)
    ly = torch.logsumexp(lp[yes_ids], 0).item(); ln = torch.logsumexp(lp[no_ids], 0).item()
    top = torch.topk(lp, 5)
    gen = out.sequences[0, inputs["input_ids"].shape[1]:]
    return dict(grid=grid.tolist() if grid is not None else None, image_tokens=n_img_tok, prompt_tokens=int(inputs["input_ids"].shape[1]),
                raw=tok.decode(gen, skip_special_tokens=True), n_gen=int(gen.shape[0]), z=round(ly - ln, 3),
                coverage=round(math.exp(ly) + math.exp(ln), 4), top5=[(tok.decode([int(i)]), round(math.exp(float(v)), 4)) for v, i in zip(top.values, top.indices)],
                seconds=round(time.time() - t, 1))

for name in picks:
    img = Image.open(img_dir / name).convert("L").convert("RGB")
    for pid in ("P0", "P1", "P2hi"):
        emit(stage="item", image=name, view=str(meta.loc[name, "View Position"]), prompt=pid, **run(img, pid))
for pid in ("P0", "P1", "P2hi"):
    emit(stage="item", image="BLIND-TEXT", prompt=pid, **run(None, pid))
emit(stage="done", total_seconds=round(time.time() - t0, 1))
