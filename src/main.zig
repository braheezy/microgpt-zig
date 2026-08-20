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

    std.debug.print("number of names: {d}\n", .{num_docs});
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
