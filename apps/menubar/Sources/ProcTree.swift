import Foundation
import Darwin

/// Process operations run away from the UI and never search by a global name.
enum ProcTree {
    enum StopError: LocalizedError {
        case unverified, didNotExit
        var errorDescription: String? {
            switch self {
            case .unverified: return "No se pudo verificar la instancia que se quiere detener."
            case .didNotExit: return "El proceso no terminó. Revisa el diagnóstico antes de iniciar otra instancia."
            }
        }
    }

    static func portAvailable(_ port: Int) async -> Bool {
        await Task.detached(priority: .utility) {
            guard (1...65535).contains(port) else { return false }
            let descriptor = socket(AF_INET, SOCK_STREAM, 0)
            guard descriptor >= 0 else { return false }
            defer { close(descriptor) }
            var reuse: Int32 = 1
            setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            address.sin_family = sa_family_t(AF_INET)
            address.sin_port = UInt16(port).bigEndian
            address.sin_addr.s_addr = inet_addr("127.0.0.1")
            return withUnsafePointer(to: &address) { pointer in
                pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
                        && listen(descriptor, 1) == 0
                }
            }
        }.value
    }

    /// Only used for a process Studio launched through its setsid/exec wrapper.
    static func terminateOwned(_ process: Process) async -> Bool {
        await Task.detached(priority: .utility) {
            let pid = process.processIdentifier
            guard pid > 1 else { return false }
            // The session leader may already have exited while a native child
            // still belongs to its group. The group ID is this owned PID.
            var group = getpgid(pid) == pid || exists(-pid)
            if group { kill(-pid, SIGTERM) }
            else if process.isRunning { kill(pid, SIGTERM) }
            let deadline = Date().addingTimeInterval(6)
            while Date() < deadline {
                group = group || getpgid(pid) == pid || exists(-pid)
                if !process.isRunning && (!group || !exists(-pid)) { return true }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            if group { kill(-pid, SIGKILL) }
            if process.isRunning { kill(pid, SIGKILL) }
            let forcedDeadline = Date().addingTimeInterval(2)
            while Date() < forcedDeadline {
                if !process.isRunning && (!group || !exists(-pid)) { return true }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
            return !process.isRunning && (!group || !exists(-pid))
        }.value
    }

    /// Explicit CLI stop verifies the server identity before signalling only
    /// its captured descendants. GUI disconnect never calls this operation.
    static func terminateServer(pid: Int32, port: Int) async throws {
        let status = try await StatusClient().status(port: port)
        guard status.pid == pid, status.port == port else { throw StopError.unverified }
        try await Task.detached(priority: .utility) {
            let listeners = command("/usr/sbin/lsof", ["-n", "-P", "-a", "-p", String(pid), "-iTCP:\(port)", "-sTCP:LISTEN", "-t"])
                .split(whereSeparator: { $0.isWhitespace })
            guard let root = identity(pid), root.uid == getuid(),
                  root.command.contains("-m server.server"), listeners.contains(Substring(String(pid)))
            else { throw StopError.unverified }
            var targets = descendants(of: pid) + [root]
            signal(root, SIGTERM)
            let deadline = Date().addingTimeInterval(6)
            while Date() < deadline {
                if identity(pid) == root {
                    for child in descendants(of: pid) where !targets.contains(child) { targets.append(child) }
                }
                if targets.allSatisfy({ identity($0.pid) != $0 }) { return }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            for target in targets { signal(target, SIGKILL) }
            let forcedDeadline = Date().addingTimeInterval(2)
            while Date() < forcedDeadline {
                if targets.allSatisfy({ identity($0.pid) != $0 }) { return }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            throw StopError.didNotExit
        }.value
    }

    private struct Identity: Equatable {
        let pid: Int32
        let uid: uid_t
        let started: String
        let command: String
    }

    private static func exists(_ pid: Int32) -> Bool { kill(pid, 0) == 0 || errno == EPERM }

    private static func identity(_ pid: Int32) -> Identity? {
        guard pid > 1 else { return nil }
        let output = command("/bin/ps", ["-p", String(pid), "-o", "uid=,lstart=,command="])
        let fields = output.split(maxSplits: 6, whereSeparator: { $0.isWhitespace })
        guard fields.count == 7, let uid = UInt32(fields[0]) else { return nil }
        return Identity(pid: pid, uid: uid, started: fields[1...5].joined(separator: " "), command: String(fields[6]))
    }

    private static func descendants(of pid: Int32) -> [Identity] {
        let children = command("/usr/bin/pgrep", ["-P", String(pid)])
            .split(whereSeparator: { $0.isWhitespace }).compactMap { Int32($0) }
        return children.flatMap { child -> [Identity] in
            guard let current = identity(child), current.uid == getuid() else { return [] }
            return descendants(of: child) + [current]
        }
    }

    private static func signal(_ target: Identity, _ number: Int32) {
        if identity(target.pid) == target { kill(target.pid, number) }
    }

    private static func command(_ executable: String, _ arguments: [String]) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }
}
