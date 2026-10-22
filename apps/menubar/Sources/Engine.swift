import Foundation

enum EngineState {
    case stopped
    case starting
    case running
}

enum EngineError: LocalizedError {
    case modelMissing(String)
    case venvMissing(String)
    case alreadyRunning

    var errorDescription: String? {
        switch self {
        case let .modelMissing(model):
            return "Modelo no instalado: \(model)"
        case let .venvMissing(root):
            return "Falta el .venv en \(root)"
        case .alreadyRunning:
            return "Ya hay un engine en marcha"
        }
    }
}

/// Spawns and watches the splash server; talks to it over HTTP.
final class Engine {
    private(set) var state: EngineState = .stopped
    private var process: Process?
    private var pollTask: Task<Void, Never>?
    /// Last /status payload, decoded loosely so engine version drift does not break the app.
    private(set) var lastStatus: [String: Any]?

    var port: Int = 8001
    var onChange: (() -> Void)?

    // MARK: Models

    /// The installed models under <root>/install/models, as OWNER/REPO[:VARIANT].
    func models(root: String) -> [(id: String, sizeGB: Double)] {
        let fm = FileManager.default
        let base = (root as NSString).appendingPathComponent("install/models")
        var found: [(id: String, sizeGB: Double)] = []
        guard let owners = try? fm.contentsOfDirectory(atPath: base) else { return found }
        for owner in owners.sorted() {
            guard !owner.hasPrefix(".") else { continue }
            let ownerPath = base + "/" + owner
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: ownerPath, isDirectory: &isDir), isDir.boolValue,
                  let repos = try? fm.contentsOfDirectory(atPath: ownerPath) else { continue }
            for repo in repos.sorted() {
                guard !repo.hasPrefix(".") else { continue }
                let dir = ownerPath + "/" + repo
                fm.fileExists(atPath: dir, isDirectory: &isDir)
                guard isDir.boolValue else { continue }
                found.append(("\(owner)/\(repo)", Self.sizeGB(of: dir)))
            }
        }
        return found
    }

    /// Apparent size of a (possibly symlinked) directory, following links.
    static func sizeGB(of path: String) -> Double {
        let fm = FileManager.default
        let real = (path as NSString).resolvingSymlinksInPath
        var total = 0.0
        var count = 0
        if let en = fm.enumerator(atPath: real) {
            while let rel = en.nextObject() as? String, count < 20_000 {
                count += 1
                // The files are symlinks into the Hugging Face blob store;
                // resolve them so the size is the target's, not the link's.
                let url = URL(fileURLWithPath: real).appendingPathComponent(rel)
                let resolved = (url.path as NSString).resolvingSymlinksInPath
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: resolved, isDirectory: &isDir), !isDir.boolValue else { continue }
                if let attrs = try? fm.attributesOfItem(atPath: resolved),
                   let size = attrs[.size] as? Int64 {
                    total += Double(size)
                }
            }
        }
        return total / 1_000_000_000
    }

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

        // Fresh engine log for this run.
        let logURL = Config.self.logFileURL()
        try? FileManager.default.createDirectory(
            at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? Data().write(to: logURL)
        let header = "splash-menubar " + Date().description + " " + python + " "
            + argv.joined(separator: " ") + "\n"
        try? Data(header.utf8).append(to: logURL)

        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = argv
        p.currentDirectoryURL = URL(fileURLWithPath: root)
        let fh = try FileHandle(forWritingTo: logURL)
        p.standardOutput = fh
        p.standardError = fh
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self, self.process === p else { return }
                self.process = nil
                self.state = .stopped
                self.lastStatus = nil
                self.pollTask?.cancel()
                self.onChange?()
            }
        }
        try p.run()
        process = p
        state = .starting
        onChange?()
        startPolling()
    }

    func stop() {
        pollTask?.cancel()
        if let p = process, p.isRunning {
            p.terminate()
            let deadline = Date().addingTimeInterval(8)
            while p.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.2)
            }
            if p.isRunning {
                kill(p.processIdentifier, SIGKILL)
            }
        }
        // The engine binary is a child of python; make sure it went too.
        // The pattern matches this repo's build/ only (never the Bionic vendor copy).
        Self.runShell("/bin/pkill", ["-f", "build/splash serve-native"])
        process = nil
        state = .stopped
        lastStatus = nil
        onChange?()
    }

    /// Runs a short-lived helper command (used for the pkill sweep).
    private static func runShell(_ launchPath: String, _ args: [String]) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launchPath)
        p.arguments = args
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        try? p.run()
        p.waitUntilExit()
    }

    // MARK: Polling

    func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self else { return }
                let port = self.port
                let state = await MainActor.run { self.state }
                guard state != .stopped else { continue }
                let alive = await MainActor.run { self.process?.isRunning ?? false }
                if state == .starting && !alive {
                    await MainActor.run {
                        self.state = .stopped
                        self.lastStatus = nil
                        self.onChange?()
                    }
                    continue
                }
                // The sync HTTP helpers wait on a semaphore, so keep them off
                // the main thread.
                let ready = state == .starting
                    ? await Task.detached(priority: .utility) { self.isReadySync(port) }.value
                    : true
                let status = await Task.detached(priority: .utility) { self.statusSync(port) }.value
                await MainActor.run {
                    if state == .starting, ready {
                        self.state = .running
                    }
                    self.lastStatus = status
                    self.onChange?()
                }
            }
        }
    }

    // MARK: HTTP

    /// GET /status; nil when the server answers nothing usable.
    func statusSync(_ port: Int) -> [String: Any]? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/status") else { return nil }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2.0
        let sem = DispatchSemaphore(value: 0)
        var result: [String: Any]?
        URLSession.shared.dataTask(with: req) { data, _, _ in
            defer { sem.signal() }
            guard let data else { return }
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                result = obj
            }
        }.resume()
        _ = sem.wait(timeout: .now() + 2.5)
        return result
    }

    func isReadySync(_ port: Int) -> Bool {
        guard let url = URL(string: "http://127.0.0.1:\(port)/ready") else { return false }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2.0
        let sem = DispatchSemaphore(value: 0)
        var ok = false
        URLSession.shared.dataTask(with: req) { _, response, _ in
            defer { sem.signal() }
            ok = (response as? HTTPURLResponse)?.statusCode == 200
        }.resume()
        _ = sem.wait(timeout: .now() + 2.5)
        return ok
    }

    /// POST /v1/memory/release: ask the engine to free its weights now.
    func releaseMemory(_ port: Int) {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/memory/release") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = Data("{}".utf8)
        URLSession.shared.dataTask(with: req) { _, _, _ in }.resume()
    }
}

extension Data {
    /// Appends these bytes to the file at `url`, creating it if needed.
    func append(to url: URL) throws {
        if FileManager.default.fileExists(atPath: url.path),
           let fh = try? FileHandle(forWritingTo: url) {
            fh.seekToEndOfFile()
            try fh.write(contentsOf: self)
            try? fh.close()
        } else {
            try write(to: url)
        }
    }
}

extension Config {
    static func logFileURL() -> URL {
        FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/splash-menubar/logs/engine.log")
    }
}
