import SwiftUI

// MARK: - Palette (MTPLX "hippo": dark base, green hero, orange warnings)

extension Color {
    static let bg = Color(red: 0.059, green: 0.078, blue: 0.098)
    static let card = Color.white.opacity(0.05)
    static let cardStroke = Color.white.opacity(0.09)
    static let label = Color.white.opacity(0.42)
    static let value = Color.white.opacity(0.95)
    static let sub = Color.white.opacity(0.55)
    static let accent = Color(red: 0.0, green: 0.84, blue: 0.56)    // #00d68f
    static let warn = Color(red: 0.98, green: 0.61, blue: 0.0)      // #f99c00
    static let danger = Color(red: 0.94, green: 0.35, blue: 0.42)   // #f0586a
    static let info = Color(red: 0.31, green: 0.71, blue: 0.95)     // #4fb6f3
}

// MARK: - Cards

struct MetricCard: View {
    let title: String
    let main: String
    var sub: String = ""
    var color: Color = .value

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title.uppercased())
                .font(.system(size: 9.5, weight: .semibold))
                .tracking(0.8)
                .foregroundStyle(Color.label)
            Text(main)
                .font(.system(size: 17, weight: .semibold).monospacedDigit())
                .foregroundStyle(color)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            Text(sub)
                .font(.system(size: 10).monospacedDigit())
                .foregroundStyle(Color.sub)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, minHeight: 62, alignment: .topLeading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.card))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.cardStroke, lineWidth: 1))
    }
}

struct ContextCard: View {
    let used: Int
    let total: Int

    private var pct: Double { total > 0 ? min(1, Double(used) / Double(total)) : 0 }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("CONTEXTO")
                    .font(.system(size: 9.5, weight: .semibold))
                    .tracking(0.8)
                    .foregroundStyle(Color.label)
                Spacer()
                Text(used == 0 ? "libre" :
                     "\(Fmt.tok(used)) / \(Fmt.tok(total)) · \(Int(pct * 100))%")
                    .font(.system(size: 10).monospacedDigit())
                    .foregroundStyle(Color.sub)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.08))
                    Capsule()
                        .fill(LinearGradient(colors: [.accent, .info],
                                             startPoint: .leading, endPoint: .trailing))
                        .frame(width: Swift.max(8, geo.size.width * pct))
                }
            }
            .frame(height: 8)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.card))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.cardStroke, lineWidth: 1))
    }
}

// MARK: - Live panel

struct LivePanel: View {
    let snapshot: Snapshot?

    var body: some View {
        Grid(horizontalSpacing: 10, verticalSpacing: 10) {
            GridRow {
                card("decode",
                     main: snapshot?.decodeTps.map { String(format: "%.1f tok/s", $0) } ?? "—",
                     sub: "ITL " + (snapshot?.itlMs.map(Fmt.ms) ?? "—"),
                     color: (snapshot?.decodeTps ?? 0) > 0 ? .accent : .value)
                card("prefill",
                     main: snapshot?.prefillTps.map { String(format: "%.0f tok/s", $0) } ?? "—",
                     sub: "TTFT " + (snapshot?.ttftMs.map(Fmt.ms) ?? "—"),
                     color: .value)
                card("draft",
                     main: snapshot?.draftPct.map { String(format: "%.1f%%", $0) } ?? "off",
                     sub: draftSub,
                     color: .info)
            }
            GridRow {
                card("requests",
                     main: "\(snapshot?.submitted ?? 0)",
                     sub: "\(snapshot?.completed ?? 0) ok · \(snapshot?.cancelled ?? 0) cancel · \(snapshot?.failed ?? 0) fallo")
                card("kv cache",
                     main: snapshot?.kvBytes.map(Fmt.bytes) ?? "—",
                     sub: kvSub)
                card("prefix cache",
                     main: snapshot?.cacheHit.map { $0 > 0 ? String(format: "%.1f%%", $0) : "—" } ?? "—",
                     sub: "reusados " + (snapshot?.cacheReused.map(Fmt.tokD) ?? "0"))
            }
            GridRow {
                card("memoria",
                     main: snapshot?.memAlloc.map(Fmt.bytes) ?? "—",
                     sub: memSub,
                     color: memColor)
                card("térmico",
                     main: snapshot?.thermal.map { $0 == "nominal" ? "nominal" : $0 } ?? "—",
                     sub: [snapshot?.device, snapshot?.ramGB.map { "\($0) GB" }].compactMap { $0 }.joined(separator: " · "),
                     color: thermalColor)
                card("weights",
                     main: snapshot?.weightsState ?? "—",
                     sub: snapshot?.weightsIdleMin.map { "liberan en $\(minToDisplay($0))" } ?? "idle off")
            }
            GridRow {
                ContextCard(used: snapshot?.ctxUsed ?? 0, total: snapshot?.ctxMax ?? 0)
                    .gridCellColumns(3)
            }
        }
    }

    private func card(_ title: String, main: String, sub: String = "", color: Color = .value) -> some View {
        MetricCard(title: title, main: main, sub: sub, color: color)
    }

    private var draftSub: String {
        guard let a = snapshot?.draftAccepted else { return "sin draft" }
        return "aceptados " + Fmt.tokD(a)
    }
    private var kvSub: String {
        guard let a = snapshot?.kvActivePages else { return "—" }
        let t = snapshot?.kvTotalPages.map(Fmt.tok) ?? "?"
        return "\(Fmt.tok(a)) / \(t) páginas"
    }
    private var memSub: String {
        let limit = snapshot?.memLimit.map { "límite " + Fmt.bytes($0) } ?? ""
        let pressure = snapshot?.pressure.map { "presión \($0)" } ?? ""
        return [limit, pressure].filter { !$0.isEmpty }.joined(separator: " · ")
    }
    private var memColor: Color {
        guard let p = snapshot?.pressure else { return .value }
        switch p {
        case "normal": return .value
        case "warn": return .warn
        default: return .danger
        }
    }
    private var thermalColor: Color {
        guard let t = snapshot?.thermal else { return .value }
        return t == "nominal" ? .accent : (t == "fair" ? .warn : .danger)
    }
    private func minToDisplay(_ m: Int) -> String {
        m >= 60 ? String(format: "%.1f h", Double(m) / 60) : "\(m) min"
    }
}

// MARK: - In-flight requests

struct InFlightPanel: View {
    let requests: [InFlight]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if requests.isEmpty {
                Text("sin peticiones en curso")
                    .font(.system(size: 11).monospaced())
                    .foregroundStyle(Color.label)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 4)
            } else {
                ForEach(requests.prefix(6)) { f in
                    Text(InFlight.line(f))
                        .font(.system(size: 11).monospaced())
                        .foregroundStyle(Color.value.opacity(0.85))
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                if requests.count > 6 {
                    Text("+\(requests.count - 6) más")
                        .font(.system(size: 11).monospaced())
                        .foregroundStyle(Color.label)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.card))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.cardStroke, lineWidth: 1))
    }
}
