const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const MAX_DOCS = 85000;
const MAX_DOC_LEN = 512;
const MAX_CHARS = 128;
var docs: [MAX_DOCS][MAX_DOC_LEN]u8 = undefined;
var num_docs: usize = 0;

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
