import Foundation

// MARK: - Formatting

enum Fmt {
    static func bytes(_ n: Double) -> String {
        guard n.isFinite, n >= 0 else { return "—" }
        if n >= 1_073_741_824 { return String(format: "%.1f GiB", n / 1_073_741_824) }
        if n >= 1_048_576 { return String(format: "%.0f MiB", n / 1_048_576) }
        if n >= 1024 { return String(format: "%.0f KiB", n / 1024) }
        return String(format: "%.0f B", n)
    }

    static func tok(_ n: Int) -> String {
        guard n >= 0 else { return "—" }
        return n >= 1_000_000 ? String(format: "%.1fM", Double(n) / 1_000_000)
            : n >= 1_000 ? String(format: "%.1fk", Double(n) / 1_000)
            : "\(n)"
    }

    static func tokD(_ n: Double) -> String {
        guard n.isFinite, n >= 0 else { return "—" }
        guard n < Double(Int.max) else { return String(format: "%.1e", n) }
        return tok(Int(n))
    }

    static func ms(_ n: Double) -> String {
        guard n.isFinite, n >= 0 else { return "—" }
        return n >= 1000 ? String(format: "%.1f s", n / 1000) : String(format: "%.0f ms", n)
    }
}

// MARK: - View data

struct InFlight: Identifiable, Sendable {
    var id: Int
    var phase: String
    var prompt: Int
    var processed: Int
    var generated: Int
    var maxNew: Int
    var age: Int

    static func line(_ f: InFlight) -> String {
        let progress: String
        switch f.phase {
        case "prefill": progress = "preparación \(Fmt.tok(f.processed))/\(Fmt.tok(f.prompt))"
        case "decode":
            progress = f.maxNew > 0
                ? "generación \(Fmt.tok(f.generated))/\(Fmt.tok(f.maxNew))"
                : "generación \(Fmt.tok(f.generated))"
        default: progress = f.phase.replacingOccurrences(of: "_", with: " ")
        }
        return "#\(f.id) · entrada \(Fmt.tok(f.prompt)) · \(progress) · \(f.age)s"
    }
}

/// Missing measurements remain unavailable; availability flags distinguish
/// missing sections from a measured zero in the compatible integer fields.
struct Snapshot: Sendable {
    var schemaVersion: Int?
    var schemaSupported = false
    var isValid = false
    var instanceID: String?
    var pid: Int32?
    var model: String = "—"
    var port: Int = 0
    var ready = false

    var decodeTps: Double?
    var liveDecodeTps: Double?
    var itlMs: Double?
    var prefillTps: Double?
    var livePrefillTps: Double?
    var ttftMs: Double?
    var draftPct: Double?
    var draftAccepted: Double?
    var draftDrafted: Double?

    var hasRequestMetrics = false
    var submitted = 0
    var completed = 0
    var cancelled = 0
    var failed = 0

    var hasSchedulerMetrics = false
    var schedDecode = 0
    var schedPrefill = 0
    var schedQueue = 0

    var kvBytes: Double?
    var kvActivePages: Int?
    var kvCachePages: Int?
    var kvFreePages: Int?
    var kvTotalPages: Int?
    /// Global resident KV capacity, including reusable prefix-cache pages.
    var kvResidentTokens: Int?
    /// Compatibility alias for global resident KV capacity, not conversation size.
    var ctxUsed: Int?
    var ctxMax: Int?
    var poolTokens: Int?

    var cacheHit: Double? // Percentage, 0...100, converted from the wire fraction.
    var cacheReused: Double?
    var ssdUsed: Double?
    var ssdCap: Double?
    var ssdDemotions: Int?
    var ssdRestores: Int?
    var diskPersistent: Bool?

    var memAlloc: Double?
    var memLimit: Double?
    var pressure: String?
    var thermal: String?
    var metalHealthy: Bool?
    var metalFailureReason: String?
    var weightsState: String?
    var weightsIdleMin: Int?
    var device: String?
    var ramGB: Int? // Legacy property name; physical RAM is expressed in GiB.
    var planCtx: Int?
    var startedAt: Double?
    var aneShare: Double?
    var aneState: String?
    var aneReason: String?

    var hasActiveRequests = false
    var inFlight: [InFlight] = []

    /// Report only explicit health evidence. A false readiness flag alone
    /// does not distinguish startup from other unavailability.
    var unavailableReason: String? {
        guard !ready else { return nil }
        if metalHealthy == false {
            let detail = metalFailureReason?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return detail.isEmpty ? "El motor informó un error de GPU."
                : "El motor informó un error de GPU: \(detail)"
        }
        if pressure == "critical" {
            return "El motor no está disponible por presión crítica de memoria."
        }
        return nil
    }
}

enum StatusDecodingError: LocalizedError {
    case unsupportedSchema(Int)
    case invalidData

    var errorDescription: String? {
        switch self {
        case .unsupportedSchema(let version):
            return "El motor usa una versión de estado no compatible (\(version))."
        case .invalidData:
            return "El motor devolvió datos de estado no válidos."
        }
    }
}

// MARK: - Typed status protocol

/// These DTOs mirror runtime/engine/Status.cpp and the HTTP instance envelope.
/// Unknown keys are ignored; missing optional metrics never become measurements.
private struct StatusDocument: Decodable {
    let schemaVersion: Int
    let ready: Bool
    let instance: Instance?
    let maximumContextTokens: Int?
    let memoryPressure: String?
    let thermalState: String?
    let metrics: Metrics?
    let requests: Requests?
    let scheduler: Scheduler?
    let kv: KV?
    let cache: Cache?
    let disk: Disk?
    let memoryActual: MemoryActual?
    let memoryGovernor: MemoryGovernor?
    let metal: MetalHealth?
    let weights: Weights?
    let aneFfn: ANE?
    let memoryPlan: MemoryPlan?
    let activeRequests: [ActiveRequest]?

    struct Instance: Decodable {
        let id: String?
        let pid: Int?
        let model: String?
        let port: Int?
        let startedAt: Double?
    }
    struct Metrics: Decodable {
        let decodeTokensPerSecond: Double?
        let prefillTokensPerSecond: Double?
        let itlMs: Latency?
        let ttftMs: Latency?
        let draftedTokens: Double?
        let acceptedDraftTokens: Double?
        let currentDecodeBatch: Batch?
        let currentPrefillBatch: Batch?
    }
    struct Latency: Decodable { let p50: Double?; let samples: Int? }
    struct Batch: Decodable { let valid: Bool?; let tokensPerSecond: Double? }
    struct Requests: Decodable {
        let submitted: Int?; let completed: Int?; let cancelled: Int?; let failed: Int?
    }
    struct Scheduler: Decodable { let decoding: Int?; let prefilling: Int?; let queued: Int? }
    struct KV: Decodable {
        let blockTokens: Int?; let pagesActive: Int?; let pagesCache: Int?
        let pagesAllocated: Int?; let pagesFree: Int?; let allocatedBytes: Double?
    }
    struct Cache: Decodable { let hitRate: Double?; let reusedTokens: Double? }
    struct Disk: Decodable {
        let usedBytes: Double?; let capacityBytes: Double?
        let kvDemotions: Int?; let kvRestores: Int?; let persistent: Bool?
    }
    struct MemoryActual: Decodable { let currentBytes: Double?; let allocatedBytes: Double? }
    struct MemoryGovernor: Decodable { let limitBytes: Double? }
    struct MetalHealth: Decodable { let healthy: Bool?; let failureReason: String? }
    struct Weights: Decodable {
        let released: Bool?; let restoring: Bool?; let idleReleaseSeconds: Double?
    }
    struct ANE: Decodable { let share: Double?; let state: String?; let reason: String? }
    struct MemoryPlan: Decodable {
        let device: Device?; let model: Model?
        struct Device: Decodable { let deviceName: String?; let physicalMemoryBytes: Double? }
        struct Model: Decodable { let maximumContextTokens: Int? }
    }
    struct ActiveRequest: Decodable {
        let id: Int?; let phase: String?; let promptTokens: Int?; let promptProcessed: Int?
        let generatedTokens: Int?; let maxNewTokens: Int?; let ageMs: Double?
    }
}

enum StatusParser {
    static let supportedSchema = 6

    /// Used by transport: malformed or unknown protocols are explicit failures.
    static func parse(data: Data) throws -> Snapshot {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let document: StatusDocument
        do {
            document = try decoder.decode(StatusDocument.self, from: data)
        } catch {
            throw StatusDecodingError.invalidData
        }
        guard document.schemaVersion == supportedSchema else {
            throw StatusDecodingError.unsupportedSchema(document.schemaVersion)
        }
        return snapshot(document)
    }

    /// Compatibility entry point for callers that already have a JSON object.
    /// The returned snapshot advertises invalid data instead of inventing values.
    static func parse(_ object: [String: Any]) -> Snapshot {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object),
              let value = try? parse(data: data) else {
            var invalid = Snapshot()
            invalid.schemaVersion = object["schema_version"] as? Int
            invalid.schemaSupported = invalid.schemaVersion == supportedSchema
            return invalid
        }
        return value
    }

    private static func snapshot(_ d: StatusDocument) -> Snapshot {
        var v = Snapshot()
        v.schemaVersion = d.schemaVersion
        v.schemaSupported = true
        v.isValid = true
        v.ready = d.ready
        v.instanceID = d.instance?.id
        if let pid = d.instance?.pid, pid > 1, pid <= Int(Int32.max) { v.pid = Int32(pid) }
        v.model = d.instance?.model ?? "—"
        if let port = d.instance?.port, (1...65535).contains(port) { v.port = port }
        v.startedAt = measured(d.instance?.startedAt)
        v.ctxMax = positive(d.maximumContextTokens)
        v.pressure = d.memoryPressure
        v.thermal = d.thermalState
        v.metalHealthy = d.metal?.healthy
        v.metalFailureReason = d.metal?.failureReason

        if let m = d.metrics {
            v.decodeTps = measured(m.decodeTokensPerSecond)
            v.prefillTps = measured(m.prefillTokensPerSecond)
            if let samples = m.itlMs?.samples, samples > 0 { v.itlMs = measured(m.itlMs?.p50) }
            if let samples = m.ttftMs?.samples, samples > 0 { v.ttftMs = measured(m.ttftMs?.p50) }
            v.draftDrafted = measured(m.draftedTokens)
            v.draftAccepted = measured(m.acceptedDraftTokens)
            if let drafted = v.draftDrafted, let accepted = v.draftAccepted,
               drafted > 0, accepted <= drafted { v.draftPct = accepted / drafted * 100 }
            if m.currentDecodeBatch?.valid == true {
                v.liveDecodeTps = measured(m.currentDecodeBatch?.tokensPerSecond)
            }
            if m.currentPrefillBatch?.valid == true {
                v.livePrefillTps = measured(m.currentPrefillBatch?.tokensPerSecond)
            }
        }
        if let r = d.requests,
           let submitted = nonnegative(r.submitted), let completed = nonnegative(r.completed),
           let cancelled = nonnegative(r.cancelled), let failed = nonnegative(r.failed) {
            v.hasRequestMetrics = true
            v.submitted = submitted; v.completed = completed
            v.cancelled = cancelled; v.failed = failed
        }
        if let s = d.scheduler, let decode = nonnegative(s.decoding),
           let prefill = nonnegative(s.prefilling), let queue = nonnegative(s.queued) {
            v.hasSchedulerMetrics = true
            v.schedDecode = decode; v.schedPrefill = prefill; v.schedQueue = queue
        }
        if let kv = d.kv {
            v.kvBytes = measured(kv.allocatedBytes)
            v.kvActivePages = nonnegative(kv.pagesActive)
            v.kvCachePages = nonnegative(kv.pagesCache)
            v.kvFreePages = nonnegative(kv.pagesFree)
            v.kvTotalPages = nonnegative(kv.pagesAllocated)
            // An active page can also have a prefix-cache owner. Subtracting
            // free pages counts resident physical pages without double counting.
            if let allocated = v.kvTotalPages, let free = v.kvFreePages,
               free <= allocated, let block = positive(kv.blockTokens) {
                v.kvResidentTokens = product(allocated - free, block)
            }
            v.ctxUsed = v.kvResidentTokens
            if let allocated = v.kvTotalPages, let block = positive(kv.blockTokens) {
                v.poolTokens = product(allocated, block)
            }
        }
        if let hit = measured(d.cache?.hitRate), hit <= 1 { v.cacheHit = hit * 100 }
        v.cacheReused = measured(d.cache?.reusedTokens)
        v.ssdUsed = measured(d.disk?.usedBytes)
        v.ssdCap = measured(d.disk?.capacityBytes)
        v.ssdDemotions = nonnegative(d.disk?.kvDemotions)
        v.ssdRestores = nonnegative(d.disk?.kvRestores)
        v.diskPersistent = d.disk?.persistent
        // current_bytes also includes native device allocations outside the backend.
        v.memAlloc = measured(d.memoryActual?.currentBytes) ?? measured(d.memoryActual?.allocatedBytes)
        if let limit = measured(d.memoryGovernor?.limitBytes), limit > 0 { v.memLimit = limit }
        if let weights = d.weights {
            if weights.restoring == true { v.weightsState = "restaurando" }
            else if weights.released == true { v.weightsState = "liberados" }
            else if weights.released == false && weights.restoring == false { v.weightsState = "cargados" }
            if let seconds = measured(weights.idleReleaseSeconds), seconds > 0,
               seconds / 60 < Double(Int.max) { v.weightsIdleMin = Int(seconds / 60) }
        }
        if let share = measured(d.aneFfn?.share), share <= 1 { v.aneShare = share }
        v.aneState = d.aneFfn?.state
        v.aneReason = d.aneFfn?.reason
        v.device = d.memoryPlan?.device?.deviceName
        if let bytes = measured(d.memoryPlan?.device?.physicalMemoryBytes),
           bytes / 1_073_741_824 < Double(Int.max) {
            v.ramGB = Int((bytes / 1_073_741_824).rounded())
        }
        v.planCtx = positive(d.memoryPlan?.model?.maximumContextTokens)
        if let rows = d.activeRequests {
            let parsed = rows.compactMap { row -> InFlight? in
                guard let id = nonnegative(row.id), let phase = row.phase,
                      let prompt = nonnegative(row.promptTokens), let processed = nonnegative(row.promptProcessed),
                      let generated = nonnegative(row.generatedTokens), let maxNew = nonnegative(row.maxNewTokens),
                      let age = measured(row.ageMs), age / 1000 < Double(Int.max) else { return nil }
                return InFlight(id: id, phase: phase, prompt: prompt, processed: processed,
                                generated: generated, maxNew: maxNew, age: Int(age / 1000))
            }
            v.hasActiveRequests = parsed.count == rows.count
            if v.hasActiveRequests { v.inFlight = parsed }
        }
        return v
    }

    private static func measured(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0 else { return nil }
        return value
    }
    private static func nonnegative(_ value: Int?) -> Int? {
        guard let value, value >= 0 else { return nil }
        return value
    }
    private static func positive(_ value: Int?) -> Int? {
        guard let value, value > 0 else { return nil }
        return value
    }
    private static func product(_ lhs: Int, _ rhs: Int) -> Int? {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : result
    }
}

// MARK: - Incremental log history

struct RequestDone: Identifiable, Sendable {
    /// A local row identity, not a server request ID.
    let id: String
    let input: Int
    let cached: Int
    let output: Int
    let ttft: Double
    let tps: Double
}

enum LogParser {
    static func inputTokens(_ line: String) -> Int? {
        for part in line.components(separatedBy: "·") {
            let text = part.trimmingCharacters(in: .whitespaces)
            if text.hasPrefix("input ") {
                return Int(text.dropFirst(6).trimmingCharacters(in: .whitespaces)
                    .replacingOccurrences(of: ",", with: ""))
            }
        }
        return nil
    }

    /// The tailer can supply a session/byte-offset identity. The fallback is a
    /// unique local row ID so identical completed requests do not collapse.
    static func done(_ line: String, id: String? = nil) -> RequestDone? {
        let text = line.trimmingCharacters(in: .whitespaces)
        guard text.contains(" Done ") || text.hasPrefix("Done ") else { return nil }
        let parts = text.components(separatedBy: "·").map { $0.trimmingCharacters(in: .whitespaces) }
        func number(_ value: String?) -> Double? {
            guard let value,
                  let number = Double(value.replacingOccurrences(of: ",", with: "")
                    .replacingOccurrences(of: "s", with: "")),
                  number.isFinite, number >= 0, number < Double(Int.max) else { return nil }
            return number
        }
        func keyValue(_ key: String) -> Double? {
            for part in parts where part.hasPrefix(key + " ") {
                return number(String(part.dropFirst(key.count + 1)).trimmingCharacters(in: .whitespaces))
            }
            return nil
        }
        func withUnit(_ unit: String) -> Double? {
            for part in parts where part.hasSuffix(unit) {
                return number(part.replacingOccurrences(of: unit, with: "")
                    .trimmingCharacters(in: .whitespaces))
            }
            return nil
        }
        guard let input = keyValue("input"), let output = keyValue("output"),
              let ttft = keyValue("TTFT"), let tps = withUnit(" tok/s") else { return nil }
        return RequestDone(id: id ?? UUID().uuidString,
                           input: Int(input), cached: Int(keyValue("cached") ?? 0),
                           output: Int(output), ttft: ttft, tps: tps)
    }
}
