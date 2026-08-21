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

    // var lr: f32 = 3e-3;
    // var b1: f32 = 0.9;
    // var b2: f32 = 0.999;
    // var eps: f32 = 1e-8;
    // var running_loss: f32 = 3.3;
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

        // var total_loss: f32 = 0;
        var logits: [MAX_CHARS + 1]f32 = undefined;
        var pos: usize = 0;
        while (pos < n) : (pos += 1) {
            targets[pos] = tokens[pos + 1];
            gptForward(tokens[pos], pos, &logits, &saved[pos]);
        }
    }
}

var ucharsArray: [MAX_CHARS]u8 = undefined;
var num_uchars: usize = 0;
var BOS: usize = 0;
var vocab_size: usize = 0;

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
        for (0..N_HEAD) |h| {
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

// Land O' SIMD
const Vec = @Vector(16, f32);
const Vec4 = @Vector(4, f32);

fn dot4(x: []const f32, y: []const f32) f32 {
    const xv: Vec4 = x[0..4].*;
    const yv: Vec4 = y[0..4].*;
    return @reduce(.Add, xv * yv);
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
