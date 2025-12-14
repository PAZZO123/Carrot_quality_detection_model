#!/usr/bin/env python3
"""
simple_substitution_attack.py

A compact, ready-to-run simple monoalphabetic substitution attack.
Interactive input stops when you type a single line containing only: END

Usage:
    python simple_substitution_attack.py -i cipher.txt -n 10 --iter 500 --restarts 10
or:
    python simple_substitution_attack.py
    (paste ciphertext, then on a new line type: END)

Smaller defaults are chosen so the script runs quickly for testing. Increase
--iter and --restarts for stronger searching.
"""

import argparse
import random
import re
from collections import Counter

ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EN_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
COMMON_WORDS = {"THE", "AND", "TO", "OF", "IN", "IS", "IT", "YOU", "THAT", "HE"}

def clean(s):
    return re.sub(r'[^A-Z]', '', s.upper())

def init_key_by_freq(ct):
    cnt = Counter(clean(ct))
    sorted_cipher = [c for c,_ in cnt.most_common()]
    remaining = [c for c in ALPH if c not in sorted_cipher]
    sorted_cipher += remaining
    key = {}
    used_plain = set()
    for i,c in enumerate(sorted_cipher):
        p = EN_FREQ_ORDER[i] if i < len(EN_FREQ_ORDER) else ALPH[i]
        key[c] = p
        used_plain.add(p)
    unused_plain = [p for p in ALPH if p not in used_plain]
    for c in ALPH:
        if c not in key:
            key[c] = unused_plain.pop(0)
    return key

def apply_key(text, key):
    out = []
    for ch in text:
        up = ch.upper()
        if up in ALPH:
            p = key.get(up, '?')
            out.append(p.lower() if ch.islower() else p)
        else:
            out.append(ch)
    return ''.join(out)

def score_plain(pt):
    U = pt.upper()
    s = 0
    for pair in ("TH","HE","IN","ER","AN","RE","ED","ON","ES","ST"):
        s += U.count(pair) * 2
    words = re.findall(r"[A-Z]{2,}", U)
    for w in words:
        if w in COMMON_WORDS:
            s += 8
    s -= U.count('?') * 5
    return s

def random_swap(key):
    a,b = random.sample(ALPH,2)
    nk = key.copy()
    nk[a], nk[b] = nk[b], nk[a]
    return nk

def attack(ciphertext, iterations=500, restarts=10):
    best_overall = []
    cleaned = ciphertext
    for r in range(restarts):
        key = init_key_by_freq(cleaned)
        for _ in range(random.randint(0,6)):
            key = random_swap(key)
        plain = apply_key(cleaned, key)
        score = score_plain(plain)
        best_local_key, best_local_score, best_local_plain = key, score, plain

        for i in range(iterations):
            cand_key = random_swap(key)
            cand_plain = apply_key(cleaned, cand_key)
            cand_score = score_plain(cand_plain)
            if cand_score > score:
                key, score, plain = cand_key, cand_score, cand_plain
                if score > best_local_score:
                    best_local_key, best_local_score, best_local_plain = key, score, plain

        best_overall.append((best_local_score, best_local_plain, best_local_key))

    uniq = {}
    for sc,pt,k in best_overall:
        if pt not in uniq or uniq[pt][0] < sc:
            uniq[pt] = (sc,k)
    results = sorted([(sc,pt,k) for pt,(sc,k) in uniq.items()], key=lambda x:-x[0])
    return results


def main():
    p = argparse.ArgumentParser(description="Simple substitution cipher attack")
    p.add_argument("-i","--input", help="ciphertext file (optional)")
    p.add_argument("-n","--top", type=int, default=10, help="top N results to show")
    p.add_argument("--iter", type=int, default=500, help="iterations per restart")
    p.add_argument("--restarts", type=int, default=10, help="random restarts")
    p.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    args = p.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            ct = f.read()
    else:
        print("Paste ciphertext. When finished type a line with only: END  (then press Enter)")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "END":
                break
            lines.append(line)
        ct = "\n".join(lines)

    if not ct.strip():
        print("No ciphertext provided. Exiting.")
        return

    print(f"Running simple attack (iter={args.iter}, restarts={args.restarts})...")
    results = attack(ct, iterations=args.iter, restarts=args.restarts)
    topn = min(args.top, len(results))
    print(f"\nTop {topn} candidate plaintexts:\n")
    for i in range(topn):
        sc, pt, k = results[i]
        print(f"--- Rank {i+1} (score {sc}) ---")
        print(pt)
        print()

if __name__ == '__main__':
    main()
