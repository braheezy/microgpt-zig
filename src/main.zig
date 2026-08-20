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
}

var ucharsArray: [MAX_CHARS]u8 = undefined;
var num_uchars: usize = 0;
var BOS: usize = 0;
var vocab_size: usize = 0;

fn buildTokenizer() void {
    var seen: [256]bool = undefined;
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
    std.sort.pdq(u8, &ucharsArray, {}, comptime compareChar);
    BOS = num_uchars;
    vocab_size = num_uchars + 1;
}

fn compareChar(_: void, a: u8, b: u8) bool {
    return a < b;
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
