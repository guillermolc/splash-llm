import Foundation
import Combine

enum EngineState { case stopped, starting, running }

enum EngineError: LocalizedError {
    case alreadyRunning
    case venvMissing(String)
    case modelMissing(String)

    var errorDescription: String? {
        switch self {
        case .alreadyRunning: return "El engine ya está en marcha."
        case .venvMissing(let r): return "No encuentro .venv/bin/python en \(r)."
        case .modelMissing(let m): return "El modelo \(m) no está descargado en install/models/."
        }
    }
}

/// Owns the splash engine process and its live status.
final class Engine: ObservableObject {
    static let shared = Engine()

    @Published var state: EngineState = .stopped
    @Published var snapshot: Snapshot?
    /// Last engine log line (shown while loading, so the UI is never silent).
    @Published var loadLine = ""
    /// Port the engine is expected to serve on (from config).
    var port = 8001

    private var process: Process?
    private var pollTask: Task<Void, Never>?
    private var pollMisses = 0

    // MARK: Lifecycle

    func start(root: String, model: String, s: Settings) throws {
        if state != .stopped { throw EngineError.alreadyRunning }
        let fm = FileManager.default
        let python = (root as NSString).appendingPathComponent(".venv/bin/python")
        guard fm.isExecutableFile(atPath: python) else { throw EngineError.venvMissing(root) }
        let modelRoot = (root as NSString).appendingPathComponent("install/models/" + model)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: modelRoot, isDirectory: &isDir), isDir.boolValue else {
            throw EngineError.modelMissing(model)
        }

        var argv = [
            "-m", "server.server", modelRoot,
            "--tokenizer", modelRoot + "/tokenizer",
            "--model", model,
            "--port", String(s.port),
            "--kv-format", s.kvFormat,
            "--idle-release", s.idleRelease,
        ]
        if s.context != "auto" { argv += ["--max-context", s.context] }
        if s.maxMemory != "auto" { argv += ["--max-memory", s.maxMemory] }
        if s.prefillInt8 { argv.append("--prefill-int8") }
        if !s.useANE { argv.append("--disable-ane") }
        if s.cacheDisk != "auto" { argv += ["--max-cache-disk", s.cacheDisk] }

        let logURL = Config.logFileURL()
        try? FileManager.default.createDirectory(
            at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? Data().write(to: logURL)
        let header = "splash-studio " + Date().description + " " + python + " "
            + argv.joined(separator: " ") + "\n"
        if let fh = try? FileHandle(forWritingTo: logURL) {
            try? fh.write(contentsOf: Data(header.utf8))
            try? fh.close()
        }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = argv
        p.currentDirectoryURL = URL(fileURLWithPath: root)
        let fh = try FileHandle(forWritingTo: logURL)
        try? fh.seekToEndOfFile()
        p.standardOutput = fh
        p.standardError = p.standardOutput
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self, self.process === p else { return }
                self.process = nil
                if self.state != .stopped {
                    self.state = .stopped
                    self.snapshot = nil
                    self.objectWillChange.send()
                }
            }
        }
        try p.run()
        port = s.port
        process = p
        state = .starting
        loadLine = ""
        startPolling()
    }

    /// Stops the engine. Kills the whole tree (python wrapper + native child):
    /// SIGTERM, a 6 s grace period, then SIGKILL. The splash server ignores a
    /// bare SIGTERM, so the grace period is load-bearing.
    func stop() {
        pollTask?.cancel()
        let pid: Int32?
        if let p = process, p.isRunning {
            pid = p.processIdentifier
        } else {
            pid = (rawStatus?["instance"] as? [String: Any])?["pid"].flatMap { Int32($0 as? Int ?? 0) }
        }
        process = nil
        rawStatus = nil
        state = .stopped
        snapshot = nil
        pollMisses = 0
        objectWillChange.send()
        guard let pid else { return }
        DispatchQueue.global().async {
            ProcTree.killTreeSync(pid)
        }
    }

    /// Switch model/settings on a running engine: kill the old tree *for real*
    /// (waiting for the port to be free) before spawning the new one.
    func restart(root: String, model: String, s: Settings) {
        let oldPort = port
        pollTask?.cancel()
        process = nil
        state = .starting
        snapshot = nil
        objectWillChange.send()
        let pid: Int32? = (rawStatus?["instance"] as? [String: Any])?["pid"].flatMap { Int32($0 as? Int ?? 0) }
        rawStatus = nil
        DispatchQueue.global().async { [weak self] in
            if let pid { ProcTree.killTreeSync(pid) }
            // Wait for the port to actually be released.
            for _ in 0..<40 where ProcTree.portBusy(oldPort) {
                Thread.sleep(forTimeInterval: 0.25)
            }
            DispatchQueue.main.async {
                guard let self else { return }
                // start() requires .stopped: the old tree is dead and the port
                // is free, so we are, for all intents, stopped.
                self.state = .stopped
                do {
                    try self.start(root: root, model: model, s: s)
                } catch {
                    self.state = .stopped
                    self.objectWillChange.send()
                }
            }
        }
    }

    /// Adopt an engine already serving on `port` (started by another copy).
    func adoptIfRunning() {
        guard state == .stopped, process == nil else { return }
        guard isReadySync(port) else { return }
        state = .running
        rawStatus = statusSync(port)
        snapshot = rawStatus.map(StatusParser.parse)
        objectWillChange.send()
    }

    // MARK: Polling

    /// Tails the engine log for the latest progress line.
    private func updateLoadLine() {
        guard let data = try? Data(contentsOf: Config.logFileURL()),
              let text = String(data: data, encoding: .utf8) else { return }
        for raw in text.split(whereSeparator: { $0 == "\n" }).reversed() {
            let t = raw.trimmingCharacters(in: .whitespaces)
            guard !t.isEmpty else { continue }
            if t.hasPrefix("splash-studio") || t.hasPrefix("[transformers]") { continue }
            loadLine = String(t)
            return
        }
    }

    func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let self else { return }
                let port = self.port
                let state = self.state
                guard state != .stopped else { continue }
                let alive = self.process?.isRunning ?? false
                let status = statusSync(port)
                if state == .starting {
                    // Loaded: ready to serve, or the load process died.
                    if let s = status, s["ready"] as? Bool == true { self.state = .running }
                    else if !alive { self.state = .stopped }
                } else if self.process == nil, status == nil {
                    // Adopted engine (no child here) that stopped answering.
                    self.pollMisses += 1
                    if self.pollMisses >= 5 { self.state = .stopped }
                }
                if let status { self.rawStatus = status }
                self.snapshot = status.map(StatusParser.parse)
                self.pollMisses = status == nil ? self.pollMisses : 0
                if self.state == .starting { self.updateLoadLine() }
                self.objectWillChange.send()
            }
        }
    }

    // MARK: HTTP

    /// Last raw /status dict (kept for pid lookups the Snapshot doesn't carry).
    var rawStatus: [String: Any]?

    func statusSync(_ port: Int) -> [String: Any]? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/status") else { return nil }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2.0
        let sem = DispatchSemaphore(value: 0)
        var result: [String: Any]?
        let task = URLSession.shared.dataTask(with: req) { data, _, _ in
            if let data, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                result = obj
            }
            sem.signal()
        }
        task.resume()
        sem.wait()
        return result
    }

    func isReadySync(_ port: Int) -> Bool {
        guard let st = statusSync(port) else { return false }
        return st["ready"] as? Bool == true
    }

    /// POST /release-memory (frees weights, keeps KV).
    func releaseMemory(_ port: Int) {
        guard let url = URL(string: "http://127.0.0.1:\(port)/release-memory") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 5.0
        URLSession.shared.dataTask(with: req) { _, _, _ in }.resume()
    }

    // MARK: Process tree

    // MARK: Models

    struct Model: Identifiable {
        let id: String
        let sizeGB: Double
    }

    /// Model directories under install/models/ (org/repo layout; dot-dirs
    /// skipped, per-file symlinks resolved so HF blob caches size correctly).
    func models(root: String) -> [Model] {
        let fm = FileManager.default
        let base = URL(fileURLWithPath: (root as NSString).appendingPathComponent("install/models"))
        guard let orgs = try? fm.contentsOfDirectory(
            at: base, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else {
            return []
        }
        var out: [Model] = []
        for org in orgs {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: org.path, isDirectory: &isDir), isDir.boolValue else { continue }
            guard let repos = try? fm.contentsOfDirectory(
                at: org, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else {
                continue
            }
            for repo in repos {
                var repoDir: ObjCBool = false
                guard fm.fileExists(atPath: repo.path, isDirectory: &repoDir), repoDir.boolValue else { continue }
                let bytes = directoryBytes(repo.resolvingSymlinksInPath())
                out.append(Model(id: org.lastPathComponent + "/" + repo.lastPathComponent,
                                 sizeGB: Double(bytes) / 1_000_000_000))
            }
        }
        return out.sorted { $0.id < $1.id }
    }

    private func directoryBytes(_ url: URL) -> Int64 {
        let fm = FileManager.default
        var total: Int64 = 0
        if let en = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey],
            options: [.skipsHiddenFiles]) {
            for case let f as URL in en {
                let resolved = f.resolvingSymlinksInPath()
                let values = try? resolved.resourceValues(forKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey])
                if values?.isRegularFile == true {
                    total += Int64(values?.totalFileAllocatedSize ?? 0)
                }
            }
        }
        return total
    }
}
