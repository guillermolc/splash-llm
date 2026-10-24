import Foundation
import Combine

enum EngineState: Equatable { case stopped, starting, running, stopping, failed }
enum EngineOwnership: Equatable { case none, owned, external }

enum EngineError: LocalizedError {
    case alreadyRunning, externalInstance, portBusy(Int), invalidSettings(String)
    case missing(String), stoppedUnexpectedly(Int32), stopFailed, instanceChanged

    var errorDescription: String? {
        switch self {
        case .alreadyRunning: return "Detén el motor actual antes de iniciar otro."
        case .externalInstance: return "Esta instancia se inició fuera de Studio. Puedes desconectarla; su proceso no se modificará."
        case .portBusy(let port): return "El puerto \(port) está ocupado. Conecta la instancia existente o elige otro puerto."
        case .invalidSettings(let message): return message
        case .missing(let path): return "No se encuentra un archivo necesario: \(path). Comprueba la instalación seleccionada."
        case .stoppedUnexpectedly(let code): return "El motor terminó con código \(code). Revisa el diagnóstico."
        case .stopFailed: return "El motor no terminó de detenerse. No se inició otra instancia."
        case .instanceChanged: return "La instancia del puerto cambió. Vuelve a conectar para controlarla."
        }
    }
}

/// Publishes UI state on the main actor; HTTP, logs and discovery are async.
@MainActor
final class Engine: ObservableObject {
    static let shared = Engine()

    @Published private(set) var state: EngineState = .stopped
    @Published private(set) var ownership: EngineOwnership = .none
    @Published private(set) var snapshot: Snapshot?
    @Published private(set) var loadLine = ""
    @Published private(set) var decodeSeries: [Double] = []
    @Published private(set) var recent: [RequestDone] = []
    @Published private(set) var errorMessage: String?
    @Published private(set) var lastUpdated: Date?
    @Published private(set) var appliedSettings: Settings?
    private(set) var port = 8001
    var endpoint: URL { URL(string: "http://127.0.0.1:\(port)")! }

    private let client = StatusClient()
    private let catalog = ModelCatalog()
    private let logTail = LogTail()
    private let logURL: URL
    private var process: Process?
    private var pollTask: Task<Void, Never>?
    private var stopTask: Task<Bool, Never>?
    private var generation: UInt64 = 0
    private var instanceID: String?
    private var isVisible = true
    private var misses = 0
    private var waitingSince: Date?
    private var hasBeenReady = false

    init(logURL: URL = Config.logFileURL()) { self.logURL = logURL }

    struct Model: Identifiable, Sendable {
        let id: String
        let sizeGB: Double
        var maximumContextTokens: Int? = nil
    }

    func dismissError() { errorMessage = nil }
    func setVisible(_ visible: Bool) { isVisible = visible }

    func models(root: String, force: Bool = false) async -> [Model] {
        await catalog.models(root: root, force: force)
    }

    func start(root: String, model: String, s: Settings) async throws {
        guard process == nil, ownership == .none, state == .stopped || state == .failed else {
            throw EngineError.alreadyRunning
        }
        var settings = s
        settings.splashRoot = root
        settings.model = model
        let errors = settings.validationErrors(maximumContextTokens: nil)
        guard errors.isEmpty else { throw EngineError.invalidSettings(errors.joined(separator: "\n")) }

        let token = nextGeneration()
        state = .starting
        errorMessage = nil
        snapshot = nil
        instanceID = nil
        hasBeenReady = false
        lastUpdated = nil
        loadLine = "Comprobando instalación…"
        port = settings.port
        do {
            guard await ProcTree.portAvailable(settings.port) else { throw EngineError.portBusy(settings.port) }
            let prepared = try await Self.prepare(settings, logURL: logURL)
            guard token == generation, !Task.isCancelled else {
                try? prepared.log.close()
                throw CancellationError()
            }
            await logTail.reset()
            guard token == generation, !Task.isCancelled else {
                try? prepared.log.close()
                throw CancellationError()
            }
            let child = Process()
            child.executableURL = URL(fileURLWithPath: prepared.python)
            // Foundation can already create a process-group leader. Otherwise
            // setsid isolates the group before exec preserves its PID.
            child.arguments = ["-c", "import os,sys; os.setsid() if os.getpgrp()!=os.getpid() else None; os.execv(sys.executable,[sys.executable]+sys.argv[1:])"] + prepared.arguments
            child.currentDirectoryURL = URL(fileURLWithPath: prepared.root)
            child.standardOutput = prepared.log
            child.standardError = prepared.log
            child.terminationHandler = { [weak self] terminated in
                Task { @MainActor in await self?.didExit(terminated, token: token) }
            }
            do { try child.run() }
            catch { try? prepared.log.close(); throw error }
            try? prepared.log.close()
            process = child
            ownership = .owned
            appliedSettings = settings
            decodeSeries = []
            recent = []
            waitingSince = Date()
            loadLine = "Cargando \(model)…"
            startPolling(token: token)
        } catch {
            guard token == generation else { throw CancellationError() }
            state = error is CancellationError ? .stopped : .failed
            if !(error is CancellationError) { errorMessage = error.localizedDescription }
            throw error
        }
    }

    func restart(root: String, model: String, s: Settings) async throws {
        guard ownership != .external else { throw EngineError.externalInstance }
        var candidate = s
        candidate.splashRoot = root
        candidate.model = model
        let errors = candidate.validationErrors(maximumContextTokens: nil)
        guard errors.isEmpty else { throw EngineError.invalidSettings(errors.joined(separator: "\n")) }
        let token = generation
        _ = try await Self.installation(candidate)
        guard token == generation, !Task.isCancelled else { throw CancellationError() }
        await stop()
        guard state == .stopped, process == nil else { throw EngineError.stopFailed }
        try await start(root: root, model: model, s: candidate)
    }

    func stop() async {
        if let stopTask {
            _ = await stopTask.value
            while self.stopTask != nil { await Task.yield() }
            return
        }
        let token = nextGeneration()
        let child = process
        state = .stopping
        errorMessage = nil
        if ownership == .external || child == nil {
            clearConnection()
            state = .stopped
            return
        }
        let stopping = Task { await ProcTree.terminateOwned(child!) }
        stopTask = stopping
        let exited = await stopping.value
        guard token == generation else { return }
        stopTask = nil
        guard exited else {
            state = .failed
            errorMessage = EngineError.stopFailed.localizedDescription
            return
        }
        process = nil
        clearConnection()
        state = .stopped
    }

    /// Connects to a configured endpoint without taking ownership of its PID.
    func adoptIfRunning(port: Int) async {
        guard process == nil, ownership == .none, state == .stopped || state == .failed else { return }
        guard (1...65535).contains(port) else {
            errorMessage = StatusClient.ClientError.invalidPort.localizedDescription
            state = .failed
            return
        }
        let token = nextGeneration()
        self.port = port
        do {
            let status = try await client.status(port: port)
            guard token == generation, !Task.isCancelled else { return }
            guard status.port == port, status.pid != nil, let id = status.instanceID else { throw EngineError.instanceChanged }
            ownership = .external
            appliedSettings = nil
            instanceID = id
            snapshot = status
            recent = []
            decodeSeries = []
            loadLine = ""
            lastUpdated = Date()
            hasBeenReady = status.ready
            updateReadiness(status)
            waitingSince = nil
            startPolling(token: token)
        } catch {
            guard token == generation, !Task.isCancelled else { return }
            clearConnection()
            state = .stopped
            // No listener is normal during app startup. Protocol/auth failures
            // remain visible instead of being mistaken for a stopped engine.
            if let urlError = error as? URLError,
               [.cannotConnectToHost, .networkConnectionLost, .timedOut].contains(urlError.code) { return }
            if !(error is CancellationError) { errorMessage = error.localizedDescription }
        }
    }

    func releaseMemory(_ port: Int) async throws {
        let token = generation
        do {
            guard ownership != .none, port == self.port else { throw EngineError.instanceChanged }
            let current = try await client.status(port: port)
            guard token == generation, current.instanceID == instanceID else { throw EngineError.instanceChanged }
            try await client.releaseMemory(port: port)
            let updated = try await client.status(port: port)
            guard token == generation, updated.instanceID == instanceID else { throw EngineError.instanceChanged }
            snapshot = updated
            lastUpdated = Date()
            errorMessage = nil
        } catch {
            if token == generation, !(error is CancellationError) { errorMessage = error.localizedDescription }
            throw error
        }
    }

    private func nextGeneration() -> UInt64 {
        pollTask?.cancel()
        pollTask = nil
        generation &+= 1
        misses = 0
        return generation
    }

    private func clearConnection() {
        ownership = .none
        instanceID = nil
        snapshot = nil
        lastUpdated = nil
        waitingSince = nil
        hasBeenReady = false
        loadLine = ""
        decodeSeries = []
        recent = []
    }

    private func startPolling(token: UInt64) {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, token == self.generation else { return }
                await self.poll(token: token)
                guard token == self.generation, !Task.isCancelled else { return }
                let interval: UInt64 = self.state == .starting || (self.isVisible && !(self.snapshot?.inFlight.isEmpty ?? true))
                    ? 1 : self.isVisible ? 2 : 5
                do { try await Task.sleep(nanoseconds: interval * 1_000_000_000) }
                catch { return }
            }
        }
    }

    private func poll(token: UInt64) async {
        do {
            let status = try await client.status(port: port)
            guard token == generation, !Task.isCancelled else { return }
            guard status.port == port, let id = status.instanceID, let pid = status.pid,
                  instanceID == nil || instanceID == id,
                  ownership != .owned || process?.processIdentifier == pid,
                  ownership != .owned || appliedSettings?.model.trimmingCharacters(in: .whitespacesAndNewlines) == status.model
            else {
                pollTask?.cancel()
                pollTask = nil
                lastUpdated = nil
                errorMessage = EngineError.instanceChanged.localizedDescription
                state = .failed
                return
            }
            instanceID = id
            snapshot = status
            lastUpdated = Date()
            misses = 0
            updateReadiness(status)
            let speed = status.schedDecode > 0 ? status.liveDecodeTps ?? 0 : 0
            decodeSeries.append(speed.isFinite ? speed : 0)
            if decodeSeries.count > 120 { decodeSeries.removeFirst(decodeSeries.count - 120) }
        } catch {
            guard token == generation, !Task.isCancelled else { return }
            misses += 1
            if misses >= 3, state != .starting {
                errorMessage = "No hay respuesta del motor. Los últimos datos pueden estar desactualizados."
            }
            if misses >= 5, state != .starting { state = .failed }
        }
        if ownership == .owned, isVisible || state == .starting {
            let lines = (try? await logTail.read(logURL)) ?? []
            guard token == generation, !Task.isCancelled else { return }
            if let last = lines.last { loadLine = last.text }
            for line in lines {
                if let done = LogParser.done(line.text, id: "\(token):\(line.id)"), !recent.contains(where: { $0.id == done.id }) {
                    recent.insert(done, at: 0)
                }
            }
            if recent.count > 5 { recent.removeLast(recent.count - 5) }
        }
        if let waitingSince, Date().timeIntervalSince(waitingSince) > 300, ownership == .owned, let child = process {
            errorMessage = "La carga no terminó en cinco minutos. Revisa el diagnóstico antes de reintentar."
            let exited = await ProcTree.terminateOwned(child)
            guard token == generation else { return }
            if exited { process = nil; ownership = .none }
            else { errorMessage = EngineError.stopFailed.localizedDescription }
            self.waitingSince = nil
            state = .failed
            pollTask?.cancel()
            pollTask = nil
        }
    }

    private func updateReadiness(_ status: Snapshot) {
        if status.ready {
            hasBeenReady = true
            waitingSince = nil
            state = .running
            errorMessage = nil
        } else if hasBeenReady || status.unavailableReason != nil {
            // A fresh unavailable engine still has useful memory/health data.
            // It must not be presented as initial model loading indefinitely.
            waitingSince = nil
            state = .failed
            errorMessage = status.unavailableReason ?? "El motor no está listo. Revisa el diagnóstico."
        } else {
            state = .starting
            errorMessage = nil
        }
    }

    private func didExit(_ child: Process, token: UInt64) async {
        guard token == generation, process === child else { return }
        pollTask?.cancel()
        pollTask = nil
        let lines = (try? await logTail.read(logURL)) ?? []
        guard token == generation, process === child else { return }
        state = .failed
        errorMessage = EngineError.stoppedUnexpectedly(child.terminationStatus).localizedDescription
        if let last = lines.last { loadLine = last.text }
        let exited = await ProcTree.terminateOwned(child)
        guard token == generation, process === child else { return }
        if exited { process = nil; ownership = .none }
        else { errorMessage = EngineError.stopFailed.localizedDescription }
        snapshot = nil
    }

    private struct Launch {
        let python: String
        let root: String
        let arguments: [String]
        let log: FileHandle
    }

    private struct Installation: Sendable {
        let root: String
        let python: String
        let binary: String
        let modelID: String
        let modelRoot: String
    }

    /// Check the candidate without touching the current process or its log.
    private static func installation(_ settings: Settings) async throws -> Installation {
        try await Task.detached(priority: .utility) {
            let manager = FileManager.default
            let root = (settings.splashRoot.trimmingCharacters(in: .whitespacesAndNewlines) as NSString).expandingTildeInPath
            let python = (root as NSString).appendingPathComponent(".venv/bin/python")
            let binary = (root as NSString).appendingPathComponent("build/splash")
            let modelID = settings.model.trimmingCharacters(in: .whitespacesAndNewlines)
            let model = (root as NSString).appendingPathComponent("install/models/" + modelID)
            for executable in [python, binary] where !manager.isExecutableFile(atPath: executable) { throw EngineError.missing(executable) }
            for file in [model + "/model.json", model + "/tokenizer"] where !manager.fileExists(atPath: file) { throw EngineError.missing(file) }
            return Installation(root: root, python: python, binary: binary, modelID: modelID, modelRoot: model)
        }.value
    }

    private static func prepare(_ settings: Settings, logURL: URL) async throws -> Launch {
        let installed = try await installation(settings)
        return try await Task.detached(priority: .utility) {
            let manager = FileManager.default
            let root = installed.root, python = installed.python, binary = installed.binary
            let modelID = installed.modelID, model = installed.modelRoot
            var arguments = ["-m", "server.server", model, "--tokenizer", model + "/tokenizer", "--binary", binary,
                             "--model", modelID, "--host", "127.0.0.1", "--port", String(settings.port),
                             "--kv-format", settings.kvFormat.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
                             "--idle-release", settings.idleRelease.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()]
            let context = settings.context.trimmingCharacters(in: .whitespacesAndNewlines)
            let memory = settings.maxMemory.trimmingCharacters(in: .whitespacesAndNewlines)
            let disk = settings.cacheDisk.trimmingCharacters(in: .whitespacesAndNewlines)
            if context.lowercased() != "auto" { arguments += ["--max-context", context.uppercased()] }
            if memory.lowercased() != "auto" { arguments += ["--max-memory", memory.uppercased()] }
            if settings.prefillInt8 { arguments.append("--prefill-int8") }
            if !settings.useANE { arguments.append("--disable-ane") }
            if disk.lowercased() != "auto" { arguments += ["--max-cache-disk", disk.uppercased()] }
            if settings.persistentCache { arguments.append("--persistent-cache") }
            let url = logURL
            try manager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let header = "splash-studio \(Date()) \(python) \(arguments.joined(separator: " "))\n"
            try Data(header.utf8).write(to: url, options: .atomic)
            let log = try FileHandle(forWritingTo: url)
            try log.seekToEnd()
            return Launch(python: python, root: root, arguments: arguments, log: log)
        }.value
    }
}
