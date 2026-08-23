# microgpt-zig

A port of [microgpt-c](https://github.com/vixhal-baraiya/microgpt-c) for learning purposes.

The most atomic way to train and inference a GPT in pure, dependency-free Zig.

A character-level transformer with forward pass, backprop, Adam, and sampling, in one Zig file with nothing beyond `std`. It trains on ~32k names in a couple of seconds and generates new ones.

## Usage

Run it:

```bash
zig build run
```

Example output (from M3 Mac):

```
...
step 19700 / 20000 | loss 1.7050 (avg 2.2417)
step 19800 / 20000 | loss 2.4132 (avg 2.2351)
step 19900 / 20000 | loss 2.3518 (avg 2.2409)
step 20000 / 20000 | loss 2.3132 (avg 2.2299)

inference
sample 01: karie
sample 02: brison
sample 03: jakana
sample 04: mayle
sample 05: kazan
sample 06: jani
sample 07: jana
sample 08: shana
sample 09: chanan
sample 10: javian
  zig fp32        9160255 tok/sec
```
