const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const MAX_DOCS = 85000;
const MAX_DOC_LEN = 512;
const MAX_CHARS = 128;
var docs: [MAX_DOCS][MAX_DOC_LEN]u8 = undefined;
var num_docs: usize = 0;

const N_EMBED = 16;
const N_HEAD = 4;
const N_LAYER = 1;
const BLOCK_SIZE = 16;
const HEAD_DIM = N_EMBED / N_HEAD;
const MLP_DIM = 4 * N_EMBED;

var wte: []f32 = undefined;
var d_wte: []f32 = undefined;
var wpe: []f32 = undefined;
var d_wpe: []f32 = undefined;
var lm_head: []f32 = undefined;
var d_lm_head: []f32 = undefined;
var attn_wq: [N_LAYER][]f32 = undefined;
var d_attn_wq: [N_LAYER][]f32 = undefined;
var attn_wk: [N_LAYER][]f32 = undefined;
var d_attn_wk: [N_LAYER][]f32 = undefined;
var attn_wv: [N_LAYER][]f32 = undefined;
var d_attn_wv: [N_LAYER][]f32 = undefined;
var attn_wo: [N_LAYER][]f32 = undefined;
var d_attn_wo: [N_LAYER][]f32 = undefined;
var mlp_fc1: [N_LAYER][]f32 = undefined;
var d_mlp_fc1: [N_LAYER][]f32 = undefined;
var mlp_fc2: [N_LAYER][]f32 = undefined;
var d_mlp_fc2: [N_LAYER][]f32 = undefined;

var adam_m_wte: []f32 = undefined;
var adam_v_wte: []f32 = undefined;
var adam_m_wpe: []f32 = undefined;
var adam_v_wpe: []f32 = undefined;
var adam_m_lm: []f32 = undefined;
var adam_v_lm: []f32 = undefined;
var adam_m_wq: [N_LAYER][]f32 = undefined;
var adam_v_wq: [N_LAYER][]f32 = undefined;
var adam_m_wk: [N_LAYER][]f32 = undefined;
var adam_v_wk: [N_LAYER][]f32 = undefined;
var adam_m_wv: [N_LAYER][]f32 = undefined;
var adam_v_wv: [N_LAYER][]f32 = undefined;
var adam_m_wo: [N_LAYER][]f32 = undefined;
var adam_v_wo: [N_LAYER][]f32 = undefined;
var adam_m_fc1: [N_LAYER][]f32 = undefined;
var adam_v_fc1: [N_LAYER][]f32 = undefined;
var adam_m_fc2: [N_LAYER][]f32 = undefined;
var adam_v_fc2: [N_LAYER][]f32 = undefined;

var kv_keys: [N_LAYER][BLOCK_SIZE][N_EMBED]f32 = undefined;
var kv_vals: [N_LAYER][BLOCK_SIZE][N_EMBED]f32 = undefined;
var dk_accum: [N_LAYER][BLOCK_SIZE][N_EMBED]f32 = undefined;
var dv_accum: [N_LAYER][BLOCK_SIZE][N_EMBED]f32 = undefined;

const PosActs = struct {
    x_embed: [N_EMBED]f32,
    rms_scale_init: f32,
    x_in: [N_LAYER][N_EMBED]f32,
    xn_attn: [N_LAYER][N_EMBED]f32,
    rms_scale_attn: [N_LAYER]f32,
    q: [N_LAYER][N_EMBED]f32,
    aw: [N_LAYER][N_HEAD][BLOCK_SIZE]f32,
    attn_out: [N_LAYER][N_EMBED]f32,
    x_mid: [N_LAYER][N_EMBED]f32,
    xn_mlp: [N_LAYER][N_EMBED]f32,
    rms_scale_mlp: [N_LAYER]f32,
    mlp_pre: [N_LAYER][MLP_DIM]f32,
    mlp_post: [N_LAYER][MLP_DIM]f32,
    x_out: [N_EMBED]f32,
};

var saved: [BLOCK_SIZE]PosActs = undefined;
var saved_probs: [BLOCK_SIZE][MAX_CHARS + 1]f32 = undefined;

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();

    // Accessing command line arguments:
    const args = try init.minimal.args.toSlice(arena);
    // In order to do I/O operations need an `Io` instance.
    const io = init.io;

    if (args.len > 1) {
        const filename = args[1];
        try loadDataset(io, filename);
    } else {
        try loadDataset(io, "data/names.txt");
    }

    var doc_order = try arena.alloc(usize, num_docs);
    for (0..num_docs) |i| {
        doc_order[i] = i;
    }

    shuffleInts(doc_order);
    const docs_tmp = try arena.alloc([MAX_DOC_LEN]u8, num_docs);
    for (0..num_docs) |i| {
        @memcpy(docs_tmp[i][0..], docs[doc_order[i]][0..]);
    }
    @memcpy(docs[0..num_docs], docs_tmp[0..num_docs]);
    arena.free(docs_tmp);
    arena.free(doc_order);
    std.debug.print("num docs: {d}\n", .{num_docs});
    buildTokenizer();
    std.debug.print("vocab size: {d}\n", .{vocab_size});
    try initParams(arena);

    const lr: f32 = 3e-3;
    const b1: f32 = 0.9;
    const b2: f32 = 0.999;
    const eps: f32 = 1e-8;
    var running_loss: f32 = 3.3;
    const num_steps: usize = 20000;

    for (0..num_steps) |step| {
        const doc = docs[step % num_docs];
        const doc_len = std.mem.indexOfScalar(u8, doc[0..], 0) orelse doc.len;

        var tokens: [MAX_DOC_LEN + 2]usize = undefined;
        var targets: [BLOCK_SIZE]usize = undefined;
        tokens[0] = BOS;
        for (0..doc_len) |i| {
            tokens[i + 1] = charToId(doc[i]) orelse unreachable;
        }
        tokens[doc_len + 1] = BOS;
        const n = if (BLOCK_SIZE < doc_len + 1) BLOCK_SIZE else doc_len + 1;

        @memset(std.mem.asBytes(&kv_keys), 0);
        @memset(std.mem.asBytes(&kv_vals), 0);

        var total_loss: f32 = 0;
        var logits: [MAX_CHARS + 1]f32 = undefined;
        var pos: usize = 0;
        while (pos < n) : (pos += 1) {
            targets[pos] = tokens[pos + 1];
            gptForward(tokens[pos], pos, &logits, &saved[pos]);
            softmaxForwardPrecise(&logits, vocab_size, &saved_probs[pos]);
            total_loss += -@log(saved_probs[pos][targets[pos]] + 1.0e-30);
        }
        const loss: f32 = total_loss / @as(f32, @floatFromInt(n));

        gptBackward(n, &tokens, &targets);

        {
            const es2 = vocab_size * N_EMBED;
            const ps2 = BLOCK_SIZE * N_EMBED;
            const as2 = N_EMBED * N_EMBED;
            const ms2 = MLP_DIM * N_EMBED;

            var gnorm2: f32 = 0.0;

            for (0..es2) |i| {
                gnorm2 += d_wte[i] * d_wte[i];
                gnorm2 += d_lm_head[i] * d_lm_head[i];
            }

            for (0..ps2) |i| {
                gnorm2 += d_wpe[i] * d_wpe[i];
            }

            for (0..N_LAYER) |l| {
                for (0..as2) |i| {
                    gnorm2 += d_attn_wq[l][i] * d_attn_wq[l][i];
                    gnorm2 += d_attn_wk[l][i] * d_attn_wk[l][i];
                    gnorm2 += d_attn_wv[l][i] * d_attn_wv[l][i];
                    gnorm2 += d_attn_wo[l][i] * d_attn_wo[l][i];
                }

                for (0..ms2) |i| {
                    gnorm2 += d_mlp_fc1[l][i] * d_mlp_fc1[l][i];
                    gnorm2 += d_mlp_fc2[l][i] * d_mlp_fc2[l][i];
                }
            }

            const gnorm = @sqrt(gnorm2);
            const clip: f32 = 1.0;

            if (gnorm > clip) {
                const scale = clip / gnorm;

                for (0..es2) |i| {
                    d_wte[i] *= scale;
                    d_lm_head[i] *= scale;
                }

                for (0..ps2) |i| {
                    d_wpe[i] *= scale;
                }

                for (0..N_LAYER) |l| {
                    for (0..as2) |i| {
                        d_attn_wq[l][i] *= scale;
                        d_attn_wk[l][i] *= scale;
                        d_attn_wv[l][i] *= scale;
                        d_attn_wo[l][i] *= scale;
                    }

                    for (0..ms2) |i| {
                        d_mlp_fc1[l][i] *= scale;
                        d_mlp_fc2[l][i] *= scale;
                    }
                }
            }
        }

        const lr_t = lr * 0.5 * (1.0 + @cos(std.math.pi * @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(num_steps))));
        const es = vocab_size * N_EMBED;
        const ps = BLOCK_SIZE * N_EMBED;
        const as = N_EMBED * N_EMBED;
        const ms = MLP_DIM * N_EMBED;
        adamUpdate(wte, d_wte, adam_m_wte, adam_v_wte, es, lr_t, b1, b2, eps, step);
        adamUpdate(wpe, d_wpe, adam_m_wpe, adam_v_wpe, ps, lr_t, b1, b2, eps, step);
        adamUpdate(lm_head, d_lm_head, adam_m_lm, adam_v_lm, es, lr_t, b1, b2, eps, step);
        for (0..N_LAYER) |i| {
            adamUpdate(attn_wq[i], d_attn_wq[i], adam_m_wq[i], adam_v_wq[i], as, lr_t, b1, b2, eps, step);
            adamUpdate(attn_wk[i], d_attn_wk[i], adam_m_wk[i], adam_v_wk[i], as, lr_t, b1, b2, eps, step);
            adamUpdate(attn_wv[i], d_attn_wv[i], adam_m_wv[i], adam_v_wv[i], as, lr_t, b1, b2, eps, step);
            adamUpdate(attn_wo[i], d_attn_wo[i], adam_m_wo[i], adam_v_wo[i], as, lr_t, b1, b2, eps, step);
            adamUpdate(mlp_fc1[i], d_mlp_fc1[i], adam_m_fc1[i], adam_v_fc1[i], ms, lr_t, b1, b2, eps, step);
            adamUpdate(mlp_fc2[i], d_mlp_fc2[i], adam_m_fc2[i], adam_v_fc2[i], ms, lr_t, b1, b2, eps, step);
        }

        running_loss = running_loss * 0.99 + loss * 0.01;
        if ((step + 1) % 100 == 0 or step == 0 or step == num_steps - 1) {
            std.debug.print("step {d:4} / {d:4} | loss {d:.4} (avg {d:.4})\n", .{ step + 1, num_steps, loss, running_loss });
        }
    }
    const temperature: f32 = 0.5;
    std.debug.print("\ninference\n", .{});

    try inferPackWeights(arena);
    const inv_t = 1.0 / temperature;

    for (0..10) |si| {
        var token_id = BOS;
        var buf: [BLOCK_SIZE + 1]u8 = [_]u8{0} ** (BLOCK_SIZE + 1);
        var len: usize = 0;
        var pos: usize = 0;
        while (pos < BLOCK_SIZE) : (pos += 1) {
            var logitsInfer: [LM_PAD_MAX]f32 = undefined;
            gptForwardInfer(token_id, pos, &logitsInfer);
            token_id = sampleLogits(&logitsInfer, vocab_size, lm_pad_global, inv_t);
            if (token_id == BOS) break;
            if (token_id < num_uchars) {
                buf[len] = ucharsArray[token_id];
                len += 1;
            }
        }
        std.debug.print("sample {d:0>2}: {s}\n", .{ si + 1, buf });

        @memset(std.mem.asBytes(&kv_keys), 0);
        @memset(std.mem.asBytes(&kv_vals), 0);
    }
    const N: usize = 5_000_000;

    var emitted: usize = 0;
    var tok: usize = BOS;
    var pos: usize = 0;

    const t0 = std.Io.Timestamp.now(io, .awake);

    var logits: [LM_PAD_MAX]f32 = undefined;

    for (0..N) |_| {
        if (pos >= BLOCK_SIZE) {
            pos = 0;
        }

        gptForwardInfer(tok, pos, logits[0..]);

        const next = sampleLogits(logits[0..], vocab_size, lm_pad_global, inv_t);

        if (next == BOS) {
            tok = BOS;
            pos = 0;
        } else {
            tok = next;
            pos += 1;
        }

        emitted += 1;
    }

    const t1 = std.Io.Timestamp.now(io, .awake);
    const elapsed_ns = t1.nanoseconds - t0.nanoseconds;
    const elapsed_s: f64 = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const tok_per_sec: f64 = @as(f64, @floatFromInt(emitted)) / elapsed_s;

    std.debug.print("  zig fp32 {d:14.0} tok/sec\n", .{tok_per_sec});
}

var ucharsArray: [MAX_CHARS]u8 = undefined;
var num_uchars: usize = 0;
var BOS: usize = 0;
var vocab_size: usize = 0;

fn inferPackWeights(allocator: Allocator) !void {
    const lm_pad = ((vocab_size + 3) / 4) * 4;
    lm_pad_global = lm_pad;
    inline for (0..N_LAYER) |li| {
        packT(attn_wq[li], &iw_q[li], N_EMBED, N_EMBED);
        packT(attn_wk[li], &iw_k[li], N_EMBED, N_EMBED);
        packT(attn_wv[li], &iw_v[li], N_EMBED, N_EMBED);
        packT(attn_wo[li], &iw_o[li], N_EMBED, N_EMBED);

        for (0..MLP_DIM / 16) |bi| {
            for (0..N_EMBED) |c| {
                for (0..16) |r| {
                    iw_fc1[li][bi * 256 + c * 16 + r] = mlp_fc1[li][(bi * 16 + r) * N_EMBED + c];
                }
            }
        }
        packT(mlp_fc2[li], &iw_fc2[li], N_EMBED, MLP_DIM);
    }
    @memset(std.mem.asBytes(&iw_lm), 0);
    for (0..vocab_size) |r| {
        for (0..N_EMBED) |c| {
            iw_lm[c * lm_pad + r] = lm_head[r * N_EMBED + c];
        }
    }
    try buildPretok(allocator);
}

const PreTok = struct {
    xin: [N_EMBED]f32,
    q: [N_EMBED]f32,
    k: [N_EMBED]f32,
    v: [N_EMBED]f32,
};

var pretok: []PreTok = undefined;
fn buildPretok(allocator: Allocator) !void {
    const count = vocab_size * BLOCK_SIZE;

    pretok = try allocator.alignedAlloc(PreTok, .@"64", count);

    for (0..vocab_size) |t| {
        for (0..BLOCK_SIZE) |p| {
            var x: [N_EMBED]f32 = undefined;
            var xr: [N_EMBED]f32 = undefined;
            var xn: [N_EMBED]f32 = undefined;

            for (0..N_EMBED) |i| {
                x[i] =
                    wte[t * N_EMBED + i] +
                    wpe[p * N_EMBED + i];
            }

            rmsNormInfer(x[0..], xr[0..]);
            rmsNormInfer(xr[0..], xn[0..]);

            const e = &pretok[t * BLOCK_SIZE + p];

            e.xin = xr;

            mv16Blk16(xn[0..], iw_q[0][0..], N_EMBED, e.q[0..]);
            mv16Blk16(xn[0..], iw_k[0][0..], N_EMBED, e.k[0..]);
            mv16Blk16(xn[0..], iw_v[0][0..], N_EMBED, e.v[0..]);
        }
    }
}
fn buildTokenizer() void {
    var seen: [256]bool = [_]bool{false} ** 256;
    for (docs[0..num_docs]) |doc| {
        var i: usize = 0;
        while (doc[i] != 0) : (i += 1) {
            seen[doc[i]] = true;
        }
    }
    for (0..256) |i| {
        if (seen[i]) {
            ucharsArray[num_uchars] = @as(u8, @intCast(i));
            num_uchars += 1;
        }
    }
    std.sort.pdq(
        u8,
        ucharsArray[0..num_uchars],
        {},
        comptime compareChar,
    );
    BOS = num_uchars;
    vocab_size = num_uchars + 1;
}
fn gptForward(
    token_id: usize,
    pos_id: usize,
    logits_out: []f32,
    act: *PosActs,
) void {
    var x: [N_EMBED]f32 = undefined;
    var tmp: [if (MLP_DIM > N_EMBED) MLP_DIM else N_EMBED]f32 = undefined;

    for (0..N_EMBED) |i| {
        x[i] = wte[token_id * N_EMBED + i] + wpe[pos_id * N_EMBED + i];
    }
    act.x_embed = x;

    act.rms_scale_init = rmsNormFwd(x[0..], x[0..]);

    for (0..N_LAYER) |li| {
        act.x_in[li] = x;
        var xn: [N_EMBED]f32 = undefined;
        act.rms_scale_attn[li] = rmsNormFwd(&x, &xn);
        act.xn_attn[li] = xn;

        var q: [N_EMBED]f32 = undefined;
        var k: [N_EMBED]f32 = undefined;
        var v: [N_EMBED]f32 = undefined;
        linearForward(&xn, attn_wq[li], N_EMBED, N_EMBED, &q);
        linearForward(&xn, attn_wk[li], N_EMBED, N_EMBED, &k);
        linearForward(&xn, attn_wv[li], N_EMBED, N_EMBED, &v);
        act.q[li] = q;

        kv_keys[li][pos_id] = k;
        kv_vals[li][pos_id] = v;
        const seq_len = pos_id + 1;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HEAD_DIM)));

        var ao: [N_EMBED]f32 = undefined;
        for (0..N_EMBED) |j| ao[j] = 0.0;
        inline for (0..N_HEAD) |h| {
            const hs = h * HEAD_DIM;
            var al: [BLOCK_SIZE]f32 = undefined;

            for (0..seq_len) |tt| {
                al[tt] = dot4(q[hs..][0..HEAD_DIM], kv_keys[li][tt][hs..][0..HEAD_DIM]) * scale;
            }

            var mx = al[0];
            for (1..seq_len) |tt| {
                if (al[tt] > mx) mx = al[tt];
            }

            var sm: f32 = 0.0;
            for (0..seq_len) |tt| {
                al[tt] = std.math.exp(al[tt] - mx);
                sm += al[tt];
            }
            const inv = 1.0 / sm;
            for (0..seq_len) |tt| {
                al[tt] *= inv;
                act.aw[li][h][tt] = al[tt];
            }

            for (0..HEAD_DIM) |j| {
                var s: f32 = 0.0;
                for (0..seq_len) |tt| {
                    s += al[tt] * kv_vals[li][tt][hs + j];
                }
                ao[hs + j] = s;
            }
        }
        act.attn_out[li] = ao;

        linearForward(&ao, attn_wo[li], N_EMBED, N_EMBED, &tmp);
        for (0..N_EMBED) |i| {
            x[i] = tmp[i] + act.x_in[li][i];
        }
        act.x_mid[li] = x;

        var xn_m: [N_EMBED]f32 = undefined;
        act.rms_scale_mlp[li] = rmsNormFwd(&x, &xn_m);
        act.xn_mlp[li] = xn_m;

        var h1: [MLP_DIM]f32 = undefined;
        linearForward(&xn_m, mlp_fc1[li], MLP_DIM, N_EMBED, &h1);
        act.mlp_pre[li] = h1;

        var h2: [MLP_DIM]f32 = undefined;
        for (0..MLP_DIM) |i| {
            h2[i] = if (h1[i] > 0) h1[i] * h1[i] else 0;
        }
        act.mlp_post[li] = h2;

        linearForward(&h2, mlp_fc2[li], N_EMBED, MLP_DIM, &tmp);
        for (0..N_EMBED) |i| {
            x[i] = tmp[i] + act.x_mid[li][i];
        }
    }

    act.x_out = x;

    linearForward(&x, lm_head, vocab_size, N_EMBED, logits_out);
}
fn gptForwardInfer(token_id: usize, pos_id: usize, logits_out: []f32) void {
    @setRuntimeSafety(false);
    var x: [N_EMBED]f32 = undefined;
    var xn: [N_EMBED]f32 = undefined;
    var xin: [N_EMBED]f32 = undefined;
    var q: [N_EMBED]f32 = undefined;
    var k: [N_EMBED]f32 = undefined;
    var v: [N_EMBED]f32 = undefined;
    var ao: [N_EMBED]f32 = undefined;

    inline for (0..N_LAYER) |li| {
        var xin_p: []const f32 = undefined;

        if (li == 0) {
            const pt = &pretok[token_id * BLOCK_SIZE + pos_id];

            xin_p = pt.xin[0..];
            q = pt.q;
            k = pt.k;
            v = pt.v;

            kv_keys[li][pos_id] = k;
            kv_vals[li][pos_id] = v;
        } else {
            xin = x;
            xin_p = xin[0..];

            rmsNormInfer(x[0..], xn[0..]);

            mv16Blk16(xn[0..], iw_q[li][0..], N_EMBED, q[0..]);
            mv16Blk16(xn[0..], iw_k[li][0..], N_EMBED, kv_keys[li][pos_id][0..]);
            mv16Blk16(xn[0..], iw_v[li][0..], N_EMBED, kv_vals[li][pos_id][0..]);

            k = kv_keys[li][pos_id];
            v = kv_vals[li][pos_id];
        }

        const seq_len = pos_id + 1;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HEAD_DIM)));

        @memset(ao[0..], 0);

        var scores: [BLOCK_SIZE]Vec4 = undefined;
        var mx: Vec4 = @splat(-3.0e38);

        for (0..seq_len) |tt| {
            const kt = kv_keys[li][tt];
            const sv: Vec4 = .{
                dot4(q[0..4], kt[0..4]) * scale,
                dot4(q[4..8], kt[4..8]) * scale,
                dot4(q[8..12], kt[8..12]) * scale,
                dot4(q[12..16], kt[12..16]) * scale,
            };
            scores[tt] = sv;
            mx = @max(mx, sv);
        }

        var sum: Vec4 = @splat(0.0);
        for (0..seq_len) |tt| {
            const ev = fastExp4(scores[tt] - mx);
            scores[tt] = ev;
            sum += ev;
        }

        const inv: Vec4 = @as(Vec4, @splat(1.0)) / sum;
        var ao_vec: [N_HEAD]Vec4 = .{
            @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0),
        };

        for (0..seq_len) |tt| {
            const weights = scores[tt] * inv;
            const vv = kv_vals[li][tt];
            inline for (0..N_HEAD) |h| {
                const hs = h * HEAD_DIM;
                const value: Vec4 = vv[hs..][0..HEAD_DIM].*;
                ao_vec[h] = @mulAdd(Vec4, @splat(weights[h]), value, ao_vec[h]);
            }
        }

        inline for (0..N_HEAD) |h| {
            const hs = h * HEAD_DIM;
            ao[hs..][0..HEAD_DIM].* = ao_vec[h];
        }

        var wo: [N_EMBED]f32 = undefined;
        mv16Blk16(ao[0..], iw_o[li][0..], N_EMBED, wo[0..]);

        for (0..N_EMBED) |i| {
            x[i] = wo[i] + xin_p[i];
        }

        const mlp_scale = rmsNormScale(x[0..]);

        var h1: [MLP_DIM]f32 = undefined;
        var h2: [MLP_DIM]f32 = undefined;
        var mlp_out: [N_EMBED]f32 = undefined;

        mvFc1(x[0..], iw_fc1[li][0..], h1[0..]);

        for (0..MLP_DIM / 4) |block| {
            const base = block * 4;
            const hv: Vec4 = h1[base..][0..4].*;
            const relu: Vec4 = @max(hv, @as(Vec4, @splat(0.0)));
            h2[base..][0..4].* = relu * relu;
        }

        mvFc2(h2[0..], iw_fc2[li][0..], mlp_out[0..]);

        const scale2 = mlp_scale * mlp_scale;
        for (0..N_EMBED) |i| {
            x[i] += scale2 * mlp_out[i];
        }
    }

    mvPacked(
        x[0..],
        iw_lm[0..],
        N_EMBED,
        lm_pad_global,
        lm_pad_global,
        logits_out[0..lm_pad_global],
    );
}

fn gptBackward(n: usize, tokens: []const usize, targets: []const usize) void {
    @memset(std.mem.asBytes(&dk_accum), 0);
    @memset(std.mem.asBytes(&dv_accum), 0);

    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n));
    var pos = n;
    while (pos > 0) {
        pos -= 1;

        const act = &saved[pos];
        const seq_len = pos + 1;

        var dl: [MAX_CHARS + 1]f32 = undefined;
        for (0..vocab_size) |i| {
            dl[i] = saved_probs[pos][i] - (if (i == targets[pos]) 1.0 else 0.0 * inv_n);
        }

        var dx: [N_EMBED]f32 = undefined;
        @memset(std.mem.asBytes(&dx), 0);
        linearBackwardX(lm_head, &dl, vocab_size, N_EMBED, &dx);
        linearBackwardW(&act.x_out, &dl, vocab_size, N_EMBED, d_lm_head);

        var li: usize = N_LAYER;
        while (li > 0) {
            li -= 1;

            var d_h2: [MLP_DIM]f32 = undefined;
            @memset(std.mem.asBytes(&d_h2), 0);

            linearBackwardX(
                mlp_fc2[li],
                dx[0..],
                N_EMBED,
                MLP_DIM,
                d_h2[0..],
            );

            linearBackwardW(
                act.mlp_post[li][0..],
                dx[0..],
                N_EMBED,
                MLP_DIM,
                d_mlp_fc2[li],
            );

            var d_h1: [MLP_DIM]f32 = undefined;
            for (0..MLP_DIM) |i| {
                d_h1[i] =
                    if (act.mlp_pre[li][i] > 0.0)
                        2.0 * act.mlp_pre[li][i] * d_h2[i]
                    else
                        0.0;
            }

            var d_xn_mlp: [N_EMBED]f32 = undefined;
            @memset(std.mem.asBytes(&d_xn_mlp), 0);

            linearBackwardX(
                mlp_fc1[li],
                d_h1[0..],
                MLP_DIM,
                N_EMBED,
                d_xn_mlp[0..],
            );

            linearBackwardW(
                act.xn_mlp[li][0..],
                d_h1[0..],
                MLP_DIM,
                N_EMBED,
                d_mlp_fc1[li],
            );

            var d_x_mid: [N_EMBED]f32 = undefined;
            @memset(std.mem.asBytes(&d_x_mid), 0);

            rmsNormBackward(
                act.x_mid[li][0..],
                act.rms_scale_mlp[li],
                d_xn_mlp[0..],
                N_EMBED,
                d_x_mid[0..],
            );

            for (0..N_EMBED) |i| {
                dx[i] += d_x_mid[i];
            }

            var d_ao: [N_EMBED]f32 = undefined;
            @memset(std.mem.asBytes(&d_ao), 0);

            linearBackwardX(attn_wo[li], dx[0..], N_EMBED, N_EMBED, d_ao[0..]);

            linearBackwardW(act.attn_out[li][0..], dx[0..], N_EMBED, N_EMBED, d_attn_wo[li]);

            var d_q: [N_EMBED]f32 = undefined;
            @memset(std.mem.asBytes(&d_q), 0);

            const scale: f32 =
                1.0 / @sqrt(@as(f32, @floatFromInt(HEAD_DIM)));

            for (0..N_HEAD) |h| {
                const hs = h * HEAD_DIM;

                var d_aw: [BLOCK_SIZE]f32 = undefined;
                @memset(std.mem.asBytes(&d_aw), 0);

                for (0..HEAD_DIM) |j| {
                    for (0..seq_len) |tt| {
                        d_aw[tt] +=
                            d_ao[hs + j] * kv_vals[li][tt][hs + j];

                        dv_accum[li][tt][hs + j] +=
                            act.aw[li][h][tt] * d_ao[hs + j];
                    }
                }

                var dot: f32 = 0.0;
                for (0..seq_len) |tt| {
                    dot += d_aw[tt] * act.aw[li][h][tt];
                }

                var d_al: [BLOCK_SIZE]f32 = undefined;
                for (0..seq_len) |tt| {
                    d_al[tt] = act.aw[li][h][tt] * (d_aw[tt] - dot);
                }

                for (0..seq_len) |tt| {
                    for (0..HEAD_DIM) |j| {
                        d_q[hs + j] +=
                            d_al[tt] * kv_keys[li][tt][hs + j] * scale;

                        dk_accum[li][tt][hs + j] +=
                            d_al[tt] * act.q[li][hs + j] * scale;
                    }
                }
            }

            var d_xn: [N_EMBED]f32 = undefined;
            @memset(std.mem.asBytes(&d_xn), 0);

            linearBackwardX(attn_wq[li], d_q[0..], N_EMBED, N_EMBED, d_xn[0..]);

            linearBackwardW(act.xn_attn[li][0..], d_q[0..], N_EMBED, N_EMBED, d_attn_wq[li]);

            linearBackwardX(attn_wk[li], dk_accum[li][pos][0..], N_EMBED, N_EMBED, d_xn[0..]);

            linearBackwardW(act.xn_attn[li][0..], dk_accum[li][pos][0..], N_EMBED, N_EMBED, d_attn_wk[li]);

            linearBackwardX(attn_wv[li], dv_accum[li][pos][0..], N_EMBED, N_EMBED, d_xn[0..]);

            linearBackwardW(act.xn_attn[li][0..], dv_accum[li][pos][0..], N_EMBED, N_EMBED, d_attn_wv[li]);

            var d_x_in: [N_EMBED]f32 = undefined;
            @memset(std.mem.asBytes(&d_x_in), 0);

            rmsNormBackward(act.x_in[li][0..], act.rms_scale_attn[li], d_xn[0..], N_EMBED, d_x_in[0..]);

            for (0..N_EMBED) |i| {
                dx[i] += d_x_in[i];
            }
        }

        var d_embed: [N_EMBED]f32 = undefined;
        @memset(std.mem.asBytes(&d_embed), 0);

        rmsNormBackward(
            act.x_embed[0..],
            act.rms_scale_init,
            dx[0..],
            N_EMBED,
            d_embed[0..],
        );

        const tok = tokens[pos];

        for (0..N_EMBED) |i| {
            d_wte[tok * N_EMBED + i] += d_embed[i];
            d_wpe[pos * N_EMBED + i] += d_embed[i];
        }
    }
}
fn adamUpdate(
    p: []f32,
    g: []f32,
    m: []f32,
    v: []f32,
    sz: usize,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    step: usize,
) void {
    const step_f: f32 = @floatFromInt(step + 1);

    const b1c = 1.0 - std.math.pow(f32, b1, step_f);
    const b2c = 1.0 - std.math.pow(f32, b2, step_f);

    for (0..sz) |i| {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];

        p[i] -= lr *
            (m[i] / b1c) /
            (@sqrt(v[i] / b2c) + eps);

        g[i] = 0.0;
    }
}

fn softmaxForwardPrecise(logits: []const f32, n: usize, probs: []f32) void {
    var mx = logits[0];

    for (1..n) |i| {
        if (logits[i] > mx) {
            mx = logits[i];
        }
    }

    var sum: f32 = 0.0;

    for (0..n) |i| {
        probs[i] = std.math.exp(logits[i] - mx);
        sum += probs[i];
    }

    const inv: f32 = 1.0 / sum;

    for (0..n) |i| {
        probs[i] *= inv;
    }
}

fn compareChar(_: void, a: u8, b: u8) bool {
    return a < b;
}
fn charToId(char: u8) ?usize {
    for (0..num_uchars) |i| {
        if (ucharsArray[i] == char) return i;
    }
    return null;
}

var num_params: usize = 0;
fn makeParam(al: Allocator, size: usize, stddev: f32) ![]f32 {
    const p = try al.alloc(f32, size);
    for (0..size) |i| {
        p[i] = rngGauss(0, stddev);
    }
    num_params += size;
    return p;
}
fn makeZero(al: Allocator, size: usize) ![]f32 {
    const p = try al.alloc(f32, size);
    @memset(p, 0);
    return p;
}
fn initParams(al: Allocator) !void {
    const es = vocab_size * N_EMBED;
    const ps = BLOCK_SIZE * N_EMBED;
    const as = N_EMBED * N_EMBED;
    const ms = MLP_DIM * N_EMBED;
    wte = try makeParam(al, es, 0.02);
    d_wte = try makeZero(al, es);
    adam_m_wte = try makeZero(al, es);
    adam_v_wte = try makeZero(al, es);
    wpe = try makeParam(al, ps, 0.02);
    d_wpe = try makeZero(al, ps);
    adam_m_wpe = try makeZero(al, ps);
    adam_v_wpe = try makeZero(al, ps);
    lm_head = try makeParam(al, es, 0.02);
    d_lm_head = try makeZero(al, es);
    adam_m_lm = try makeZero(al, es);
    adam_v_lm = try makeZero(al, es);
    for (0..N_LAYER) |i| {
        attn_wq[i] = try makeParam(al, as, 0.02);
        d_attn_wq[i] = try makeZero(al, as);
        adam_m_wq[i] = try makeZero(al, as);
        adam_v_wq[i] = try makeZero(al, as);
        attn_wk[i] = try makeParam(al, as, 0.02);
        d_attn_wk[i] = try makeZero(al, as);
        adam_m_wk[i] = try makeZero(al, as);
        adam_v_wk[i] = try makeZero(al, as);
        attn_wv[i] = try makeParam(al, as, 0.02);
        d_attn_wv[i] = try makeZero(al, as);
        adam_m_wv[i] = try makeZero(al, as);
        adam_v_wv[i] = try makeZero(al, as);
        attn_wo[i] = try makeParam(al, as, 0.02);
        d_attn_wo[i] = try makeZero(al, as);
        adam_m_wo[i] = try makeZero(al, as);
        adam_v_wo[i] = try makeZero(al, as);
        mlp_fc1[i] = try makeParam(al, ms, 0.02);
        d_mlp_fc1[i] = try makeZero(al, ms);
        adam_m_fc1[i] = try makeZero(al, ms);
        adam_v_fc1[i] = try makeZero(al, ms);
        mlp_fc2[i] = try makeParam(al, ms, 0.02);
        d_mlp_fc2[i] = try makeZero(al, ms);
        adam_m_fc2[i] = try makeZero(al, ms);
        adam_v_fc2[i] = try makeZero(al, ms);
    }
    std.debug.print("num params: {d}\n", .{num_params});
}
var rng_state: u64 = 42;

fn rngNext() u64 {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
fn rngUniform() f64 {
    return @as(f64, @floatFromInt((rngNext() >> 11))) * (1.0 / 9007199254740992.0);
}
fn rngGauss(mean: f32, stddev: f32) f32 {
    var uniform1: f64 = rngUniform();
    const uniform2: f64 = rngUniform();
    if (uniform1 < 1e-30) {
        uniform1 = 1e-30;
    }
    return mean + stddev * @as(f32, @floatCast(@sqrt(-2.0 * @log(uniform1)) * @cos(2.0 * std.math.pi * uniform2)));
}
fn shuffleInts(arr: []usize) void {
    var n = arr.len - 1;
    while (n > 0) : (n -= 1) {
        const j: usize = @intFromFloat(rngUniform() * @as(f64, @floatFromInt(n + 1)));
        const tmp = arr[n];
        arr[n] = arr[j];
        arr[j] = tmp;
    }
}
fn loadDataset(io: Io, filename: []const u8) !void {
    const file = try Io.Dir.cwd().openFile(io, filename, .{});
    defer file.close(io);

    var reader_buffer: [4096]u8 = undefined;
    var file_reader = file.reader(io, &reader_buffer);
    const reader = &file_reader.interface;

    while (num_docs < MAX_DOCS) {
        var line = (try reader.takeDelimiter('\n')) orelse break;

        if (line.len > 0 and line[line.len - 1] == '\r') {
            line = line[0 .. line.len - 1];
        }
        if (line.len == 0) continue;

        @memcpy(docs[num_docs][0..line.len], line);
        num_docs += 1;
    }
}

const LM_PAD_MAX = ((((MAX_CHARS + 1) + 3) / 4) * 4);
const ATTN_SCALE = 0.5;
var lm_pad_global: usize = LM_PAD_MAX;
var iw_q: [N_LAYER][N_EMBED * N_EMBED]f32 align(64) = undefined;
var iw_k: [N_LAYER][N_EMBED * N_EMBED]f32 align(64) = undefined;
var iw_v: [N_LAYER][N_EMBED * N_EMBED]f32 align(64) = undefined;
var iw_o: [N_LAYER][N_EMBED * N_EMBED]f32 align(64) = undefined;

var iw_fc1: [N_LAYER][N_EMBED * MLP_DIM]f32 align(64) = undefined;
var iw_fc2: [N_LAYER][MLP_DIM * N_EMBED]f32 align(64) = undefined;

var iw_lm: [N_EMBED * LM_PAD_MAX]f32 align(64) = undefined;

fn packT(src: []const f32, dst: []f32, nout: usize, nin: usize) void {
    for (0..nout) |r| {
        for (0..nin) |c| {
            dst[c * nout + r] = src[r * nin + c];
        }
    }
}
// Land O' SIMD
const Vec = @Vector(N_EMBED, f32);
const Vec4 = @Vector(4, f32);
const IntVec4 = @Vector(4, i32);

// Fast four-wide exponential approximation used by the C implementation's
// vfexpq. It is intentionally approximate; it is suitable for softmax where
// the logits are stabilized by subtracting their maximum first.
fn fastExp4(x: Vec4) Vec4 {
    const scale: Vec4 = @splat(12102203.1615614 * 1.4426950408);
    const bias: IntVec4 = @splat(1065353216);
    const y = x * scale;
    const i: IntVec4 = @intFromFloat(@round(y));
    return @bitCast(i + bias);
}

fn dot4(x: []const f32, y: []const f32) f32 {
    const xv: Vec4 = x[0..4].*;
    const yv: Vec4 = y[0..4].*;
    return @reduce(.Add, xv * yv);
}
fn mv16Blk16(x: []const f32, wcol: []const f32, ldw: usize, out: []f32) void {
    @setRuntimeSafety(false);
    const V = @Vector(4, f32);
    var a0: V = @splat(0.0);
    var a1: V = @splat(0.0);
    var a2: V = @splat(0.0);
    var a3: V = @splat(0.0);

    inline for (0..N_EMBED) |c| {
        const xv: V = @splat(x[c]);
        const base = c * ldw;
        const w0: V = .{ wcol[base], wcol[base + 1], wcol[base + 2], wcol[base + 3] };
        const w1: V = .{ wcol[base + 4], wcol[base + 5], wcol[base + 6], wcol[base + 7] };
        const w2: V = .{ wcol[base + 8], wcol[base + 9], wcol[base + 10], wcol[base + 11] };
        const w3: V = .{ wcol[base + 12], wcol[base + 13], wcol[base + 14], wcol[base + 15] };
        a0 = @mulAdd(V, xv, w0, a0);
        a1 = @mulAdd(V, xv, w1, a1);
        a2 = @mulAdd(V, xv, w2, a2);
        a3 = @mulAdd(V, xv, w3, a3);
    }

    out[0..4].* = a0;
    out[4..8].* = a1;
    out[8..12].* = a2;
    out[12..16].* = a3;
}
fn mvPacked(
    x: []const f32,
    wcol: []const f32,
    nin: usize,
    nout: usize,
    ldw: usize,
    out: []f32,
) void {
    mvPackedSimd(x, wcol, nin, nout, ldw, out);
}

// Matrix-vector multiply for column-packed weights:
// wcol[c * ldw + r] is the weight from input c to output r.
// Accumulating four output rows at a time maps directly to SIMD without
// requiring architecture-specific intrinsics.
fn mvPackedSimd(
    x: []const f32,
    wcol: []const f32,
    nin: usize,
    nout: usize,
    ldw: usize,
    out: []f32,
) void {
    @setRuntimeSafety(false);
    const V = @Vector(4, f32);
    var r: usize = 0;

    while (r + 4 <= nout) : (r += 4) {
        var acc: V = @splat(0.0);

        for (0..nin) |c| {
            const wv: V = .{
                wcol[c * ldw + r + 0],
                wcol[c * ldw + r + 1],
                wcol[c * ldw + r + 2],
                wcol[c * ldw + r + 3],
            };
            acc = @mulAdd(V, @as(V, @splat(x[c])), wv, acc);
        }

        out[r..][0..4].* = acc;
    }

    while (r < nout) : (r += 1) {
        var sum: f32 = 0.0;
        for (0..nin) |c| {
            sum += x[c] * wcol[c * ldw + r];
        }
        out[r] = sum;
    }
}

// iw_fc1 is packed in groups of 16 output rows. Each block has layout
// [input column][16 output values], matching inferPackWeights and C's
// mv16_blk16r4 helper.
fn mvFc1(
    x: []const f32,
    w: []const f32,
    out: []f32,
) void {
    @setRuntimeSafety(false);
    const V = @Vector(4, f32);

    for (0..MLP_DIM / 16) |block| {
        var a0: V = @splat(0.0);
        var a1: V = @splat(0.0);
        var a2: V = @splat(0.0);
        var a3: V = @splat(0.0);
        const block_base = block * 256;

        inline for (0..N_EMBED) |c| {
            const xv: V = @splat(x[c]);
            const base = block_base + c * 16;
            const w0: V = .{ w[base], w[base + 1], w[base + 2], w[base + 3] };
            const w1: V = .{ w[base + 4], w[base + 5], w[base + 6], w[base + 7] };
            const w2: V = .{ w[base + 8], w[base + 9], w[base + 10], w[base + 11] };
            const w3: V = .{ w[base + 12], w[base + 13], w[base + 14], w[base + 15] };
            a0 = @mulAdd(V, xv, w0, a0);
            a1 = @mulAdd(V, xv, w1, a1);
            a2 = @mulAdd(V, xv, w2, a2);
            a3 = @mulAdd(V, xv, w3, a3);
        }

        const dst = block * 16;
        out[dst..][0..4].* = a0;
        out[dst + 4 ..][0..4].* = a1;
        out[dst + 8 ..][0..4].* = a2;
        out[dst + 12 ..][0..4].* = a3;
    }
}

fn mvFc2(
    x: []const f32,
    w: []const f32,
    out: []f32,
) void {
    @setRuntimeSafety(false);
    const V = @Vector(4, f32);
    var a0: V = @splat(0.0);
    var a1: V = @splat(0.0);
    var a2: V = @splat(0.0);
    var a3: V = @splat(0.0);

    inline for (0..MLP_DIM) |c| {
        const xv: V = @splat(x[c]);
        const base = c * N_EMBED;
        const w0: V = .{ w[base], w[base + 1], w[base + 2], w[base + 3] };
        const w1: V = .{ w[base + 4], w[base + 5], w[base + 6], w[base + 7] };
        const w2: V = .{ w[base + 8], w[base + 9], w[base + 10], w[base + 11] };
        const w3: V = .{ w[base + 12], w[base + 13], w[base + 14], w[base + 15] };
        a0 = @mulAdd(V, xv, w0, a0);
        a1 = @mulAdd(V, xv, w1, a1);
        a2 = @mulAdd(V, xv, w2, a2);
        a3 = @mulAdd(V, xv, w3, a3);
    }

    out[0..4].* = a0;
    out[4..8].* = a1;
    out[8..12].* = a2;
    out[12..16].* = a3;
}

fn rmsNormFwd(x: []const f32, out: []f32) f32 {
    const v: Vec = x[0..16].*;
    const squares = v * v;

    const sum: f32 = @reduce(.Add, squares);
    const mean = sum / 16.0;
    const scale = 1.0 / @sqrt(mean + 1e-5);

    const result = v * @as(Vec, @splat(scale));
    out[0..16].* = result;

    return scale;
}
fn rmsNormBackward(
    x: []const f32,
    scale: f32,
    dout: []const f32,
    n: usize,
    dx: []f32,
) void {
    var dot: f32 = 0.0;

    for (0..n) |i| {
        dot += dout[i] * x[i];
    }

    const n_f: f32 = @floatFromInt(n);
    const coeff = scale * scale * scale / n_f;

    for (0..n) |i| {
        dx[i] += scale * dout[i] - coeff * x[i] * dot;
    }
}
fn rmsNormInfer(x: []const f32, out: []f32) void {
    @setRuntimeSafety(false);
    const v: Vec = x[0..N_EMBED].*;
    const squared = v * v;

    const sum: f32 = @reduce(.Add, squared);
    const mean = sum / @as(f32, @floatFromInt(N_EMBED));
    const scale: f32 = 1.0 / @sqrt(mean + 1e-5);

    const result = v * @as(Vec, @splat(scale));

    out[0..N_EMBED].* = result;
}
fn rmsNormScale(x: []const f32) f32 {
    var sum: f32 = 0.0;

    for (x) |value| {
        sum += value * value;
    }

    const mean = sum / @as(f32, @floatFromInt(x.len));
    return 1.0 / @sqrt(mean + 1e-5);
}

// Compute out = w * x, where w stores nout contiguous rows of nin weights.
fn linearForward(
    x: []const f32,
    w: []const f32,
    nout: usize,
    nin: usize,
    out: []f32,
) void {
    for (0..nout) |r| {
        const row = w[r * nin ..][0..nin];
        var sum: f32 = 0.0;
        for (x[0..nin], row) |xv, wv| {
            sum += xv * wv;
        }
        out[r] = sum;
    }
}
fn linearBackwardX(
    w: []const f32,
    dout: []const f32,
    nout: usize,
    nin: usize,
    dx: []f32,
) void {
    for (0..nin) |c| {
        var sum: f32 = 0.0;

        for (0..nout) |r| {
            sum += dout[r] * w[r * nin + c];
        }

        dx[c] += sum;
    }
}
fn linearBackwardW(
    x: []const f32,
    dout: []const f32,
    nout: usize,
    nin: usize,
    dw: []f32,
) void {
    for (0..nout) |r| {
        const dr = dout[r];
        const dwr = dw[r * nin ..][0..nin];

        for (0..nin) |c| {
            dwr[c] += dr * x[c];
        }
    }
}

fn sampleLogits(logits: []f32, n: usize, npad: usize, inv_t: f32) usize {
    var p: [LM_PAD_MAX]f32 = undefined;

    // Pad unused logits with logits[0], matching the C implementation.
    for (n..npad) |i| {
        logits[i] = logits[0];
    }

    var mx = logits[0];

    for (1..npad) |i| {
        if (logits[i] > mx) {
            mx = logits[i];
        }
    }

    mx *= inv_t;

    var sum: f32 = 0.0;

    var i_exp: usize = 0;
    while (i_exp + 4 <= npad) : (i_exp += 4) {
        const lv: Vec4 = logits[i_exp..][0..4].*;
        const ev = fastExp4(lv * @as(Vec4, @splat(inv_t)) - @as(Vec4, @splat(mx)));
        p[i_exp..][0..4].* = ev;
        sum += @reduce(.Add, ev);
    }
    while (i_exp < npad) : (i_exp += 1) {
        p[i_exp] = std.math.exp(logits[i_exp] * inv_t - mx);
        sum += p[i_exp];
    }

    // Remove the contribution from the artificial padded entries.
    sum -= @as(f32, @floatFromInt(npad - n)) * p[0];

    const random_value: f32 =
        @as(f32, @floatCast(rngUniform())) * sum;

    var cumulative: f32 = 0.0;

    for (0..n) |i| {
        cumulative += p[i];

        if (random_value < cumulative) {
            return i;
        }
    }

    return n - 1;
}
