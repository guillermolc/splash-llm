import Foundation

// MARK: - Formatting helpers

enum Fmt {
    static func bytes(_ n: Double) -> String {
        if n >= 1_000_000_000 { return String(format: "%.1f GB", n / 1_000_000_000) }
        if n >= 1_000_000 { return String(format: "%.0f MB", n / 1_000_000) }
        if n > 0 { return String(format: "%.0f KB", n / 1_000) }
        return "0"
    }

    static func tok(_ n: Int) -> String {
        n >= 1_000_000 ? String(format: "%.1fM", Double(n) / 1_000_000)
            : n >= 1_000 ? String(format: "%.1fk", Double(n) / 1_000)
            : "\(n)"
    }

    static func tokD(_ n: Double) -> String { tok(Int(n)) }

    static func ms(_ n: Double) -> String {
        n >= 1000 ? String(format: "%.1f s", n / 1000) : String(format: "%.0f ms", n)
    }
}

// MARK: - Snapshot of GET /status

/// What the UI renders. Everything is optional-ish: absent = "—".
struct InFlight: Identifiable {
    var id: Int
    var phase: String
    var prompt: Int
    var processed: Int
    var generated: Int
    var maxNew: Int
    var age: Int

    static func line(_ f: InFlight) -> String {
        let pos = f.phase == "prefill" ? "[\(f.processed)/\(f.prompt) prefill]" :
                  f.maxNew > 0 ? "[\(f.generated)/\(f.maxNew) decode]" :
                  "[\(f.generated) decode]"
        return "#\(f.id) · prompt \(Fmt.tok(f.prompt)) · \(pos) · \(f.age)s"
    }
}

struct Snapshot {
    var model: String = "—"
    var port: Int = 0
    var ready = false

    // metrics
    var decodeTps: Double?
    var itlMs: Double?
    var prefillTps: Double?
    var ttftMs: Double?
    var draftPct: Double?
    var draftAccepted: Double?
    var draftDrafted: Double?

    // request totals
    var submitted = 0
    var completed = 0
    var cancelled = 0
    var failed = 0

    // scheduler
    var schedDecode = 0
    var schedPrefill = 0
    var schedQueue = 0

    // KV
    var kvBytes: Double?
    var kvActivePages: Int?
    var kvTotalPages: Int?
    var ctxUsed: Int?
    var ctxMax: Int?

    // prefix cache / disk
    var cacheHit: Double?
    var cacheReused: Double?
    var ssdUsed: Double?
    var ssdCap: Double?

    // memory / system
    var memAlloc: Double?
    var memLimit: Double?
    var pressure: String?
    var thermal: String?

    // weights
    var weightsState: String?   // "cargados" / "liberados" / "restaurando"
    var weightsIdleMin: Int?

    // plan
    var device: String?
    var ramGB: Int?
    var planCtx: Int?

    var inFlight: [InFlight] = []
}

/// Pure parser: raw /status JSON → Snapshot.
enum StatusParser {
    static func parse(_ s: [String: Any]) -> Snapshot {
        var v = Snapshot()
        if let inst = s["instance"] as? [String: Any] {
            v.model = inst["model"] as? String ?? "—"
            v.port = inst["port"] as? Int ?? 0
        }
        v.ready = s["ready"] as? Bool ?? false

        if let m = s["metrics"] as? [String: Any] {
            v.decodeTps = m["decode_tokens_per_second"] as? Double
            v.prefillTps = m["prefill_tokens_per_second"] as? Double
            if let itl = m["itl_ms"] as? [String: Any] { v.itlMs = itl["p50"] as? Double }
            if let t = m["ttft_ms"] as? [String: Any] { v.ttftMs = t["p50"] as? Double }
            if let d = m["drafted_tokens"] as? Double, let a = m["accepted_draft_tokens"] as? Double, d > 0 {
                v.draftPct = a / d * 100
                v.draftAccepted = a
                v.draftDrafted = d
            }
        }
        if let r = s["requests"] as? [String: Any] {
            v.submitted = r["submitted"] as? Int ?? 0
            v.completed = r["completed"] as? Int ?? 0
            v.cancelled = r["cancelled"] as? Int ?? 0
            v.failed = r["failed"] as? Int ?? 0
        }
        if let sch = s["scheduler"] as? [String: Any] {
            v.schedDecode = sch["decoding"] as? Int ?? 0
            v.schedPrefill = sch["prefilling"] as? Int ?? 0
            v.schedQueue = sch["queued"] as? Int ?? 0
        }
        if let kv = s["kv"] as? [String: Any] {
            v.kvBytes = kv["allocated_bytes"] as? Double
            v.kvActivePages = kv["pages_active"] as? Int
            v.kvTotalPages = kv["pages_allocated"] as? Int
            if let a = kv["pages_active"] as? Int, let c = kv["pages_cache"] as? Int,
               let block = kv["block_tokens"] as? Int {
                v.ctxUsed = (a + c) * block
            }
        }
        v.ctxMax = s["maximum_context_tokens"] as? Int
        if let c = s["cache"] as? [String: Any] {
            v.cacheHit = c["hit_rate"] as? Double
            v.cacheReused = c["reused_tokens"] as? Double
        }
        if let d = s["disk"] as? [String: Any] {
            v.ssdUsed = d["used_bytes"] as? Double
            v.ssdCap = d["capacity_bytes"] as? Double
        }
        if let mem = s["memory_actual"] as? [String: Any] { v.memAlloc = mem["allocated_bytes"] as? Double }
        if let gov = s["memory_governor"] as? [String: Any] {
            let lim = gov["limit_bytes"] as? Double
            if let lim, lim > 0 { v.memLimit = lim }
        }
        v.pressure = s["memory_pressure"] as? String
        v.thermal = s["thermal_state"] as? String
        if let w = s["weights"] as? [String: Any] {
            if w["released"] as? Bool == true { v.weightsState = "liberados" }
            else if w["restoring"] as? Bool == true { v.weightsState = "restaurando" }
            else { v.weightsState = "cargados" }
            if let sec = w["idle_release_seconds"] as? Double, sec > 0 {
                v.weightsIdleMin = Int(sec / 60)
            }
        }
        if let plan = s["memory_plan"] as? [String: Any], let dev = plan["device"] as? [String: Any] {
            v.device = dev["device_name"] as? String
            if let b = dev["physical_memory_bytes"] as? Double { v.ramGB = Int(b / 1_000_000_000) }
            if let model = plan["model"] as? [String: Any] {
                v.planCtx = model["maximum_context_tokens"] as? Int
            }
        }
        if let rows = s["active_requests"] as? [[String: Any]] {
            v.inFlight = rows.map {
                InFlight(id: $0["id"] as? Int ?? 0,
                         phase: $0["phase"] as? String ?? "?",
                         prompt: $0["prompt_tokens"] as? Int ?? 0,
                         processed: $0["prompt_processed"] as? Int ?? 0,
                         generated: $0["generated_tokens"] as? Int ?? 0,
                         maxNew: $0["max_new_tokens"] as? Int ?? 0,
                         age: Int(($0["age_ms"] as? Double ?? 0) / 1000))
            }
        }
        return v
    }
}
