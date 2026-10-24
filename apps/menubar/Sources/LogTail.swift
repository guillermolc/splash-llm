import Foundation

/// Reads only appended bytes, with bounded work and bounded partial lines.
actor LogTail {
    struct Line: Sendable { let id: String; let text: String }
    private var inode: UInt64?
    private var offset: UInt64 = 0
    private var pending = Data()

    func reset() { inode = nil; offset = 0; pending.removeAll(keepingCapacity: false) }

    func read(_ url: URL) throws -> [Line] {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let currentInode = (attributes[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        if inode != currentInode || size < offset { offset = 0; pending.removeAll(keepingCapacity: true) }
        inode = currentInode
        guard size > offset else { return [] }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: offset)
        let data = try handle.read(upToCount: 256 * 1024) ?? Data()
        let start = offset - UInt64(pending.count)
        offset += UInt64(data.count)
        pending.append(data)
        var result: [Line] = []
        var begin = pending.startIndex
        for end in pending.indices where pending[end] == 10 {
            let text = String(decoding: pending[begin..<end], as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty && !text.hasPrefix("splash-studio") && !text.hasPrefix("[transformers]") {
                result.append(Line(id: "\(currentInode):\(start + UInt64(begin))", text: text))
            }
            begin = pending.index(after: end)
        }
        pending = Data(pending[begin...])
        if pending.count > 8192 { pending.removeAll(keepingCapacity: false) }
        return Array(result.suffix(200))
    }
}
