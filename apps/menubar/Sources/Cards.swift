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
    /// conv: tokens en la conversación. pool: tokens que caben en RAM.
    /// total: contexto máximo. Lo que de conv no cabe en pool está compactado.
    let conv: Int
    let pool: Int
    let total: Int

    @State private var pulse = false
    @State private var lastCompacted = -1

    private var compacted: Int { Swift.max(0, conv - pool) }
    private var poolFrac: Double { total > 0 ? Double(Swift.min(conv, pool)) / Double(total) : 0 }
    private var ssdFrac: Double { total > 0 ? Double(compacted) / Double(total) : 0 }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("CONTEXTO")
                    .font(.system(size: 9.5, weight: .semibold))
                    .tracking(0.8)
                    .foregroundStyle(Color.label)
                Spacer()
                if compacted > 0 {
                    Text("↺ \(Fmt.tok(compacted)) compactados")
                        .font(.system(size: 10, weight: .semibold).monospacedDigit())
                        .foregroundStyle(Color.warn)
                }
                Text(conv == 0 ? "libre" :
                     "\(Fmt.tok(conv)) / \(Fmt.tok(total)) · \(total > 0 ? Int(Double(conv) / Double(total) * 100) : 0)%")
                    .font(.system(size: 10).monospacedDigit())
                    .foregroundStyle(Color.sub)
            }
            GeometryReader { geo in
                HStack(spacing: 1) {
                    if conv > 0 {
                        Capsule()
                            .fill(LinearGradient(colors: [.accent, .info],
                                                 startPoint: .leading, endPoint: .trailing))
                            .frame(width: Swift.max(6, geo.size.width * poolFrac))
                        if compacted > 0 {
                            Capsule()
                                .fill(Color.warn.opacity(0.85))
                                .frame(width: ssdFrac * geo.size.width)
                        }
                    }
                    Capsule().fill(Color.white.opacity(0.08))
                        .frame(maxWidth: .infinity)
                }
            }
            .frame(height: 8)
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.warn.opacity(pulse ? 0.9 : 0.0), lineWidth: 2)
                    .padding(-2)
            )
            .animation(.easeOut(duration: 0.5), value: pulse)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.card))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.cardStroke, lineWidth: 1))
        .onAppear { lastCompacted = compacted }
        .onChange(of: compacted) { newValue in
            if newValue > lastCompacted {
                pulse = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { pulse = false }
            }
            lastCompacted = newValue
        }
    }
}

// MARK: - Live panel

struct LivePanel: View {
    let snapshot: Snapshot?
    let decodeSeries: [Double]

    var body: some View {
        Grid(horizontalSpacing: 10, verticalSpacing: 10) {
            GridRow {
                SpeedChartCard(decode: decodeSeries)
                    .gridCellColumns(3)
            }
            GridRow {
                card("decode",
                     main: decodeMain,
                     sub: "ITL " + (snapshot?.itlMs.map(Fmt.ms) ?? "—") + " · media " + (snapshot?.decodeTps.map { String(format: "%.0f", $0) } ?? "—"),
                     color: decodeMain != "—" ? .accent : .value)
                card("prefill",
                     main: prefillMain,
                     sub: "TTFT " + (snapshot?.ttftMs.map(Fmt.ms) ?? "—") + " · media " + (snapshot?.prefillTps.map { String(format: "%.0f", $0) } ?? "—"),
                     color: prefillMain != "—" ? .accent : .value)
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
                     sub: prefixSub)
            }
            GridRow {
                card("memoria",
                     main: snapshot?.memAlloc.map(Fmt.bytes) ?? "—",
                     sub: memSub,
                     color: memColor)
                card("térmico",
                     main: snapshot?.thermal.map { $0 == "nominal" ? "nominal" : $0 } ?? "—",
                     sub: thermalSub,
                     color: thermalColor)
                card("weights",
                     main: snapshot?.weightsState ?? "—",
                     sub: snapshot?.weightsIdleMin.map { "liberan en $\(minToDisplay($0))" } ?? "idle off")
            }
            GridRow {
                ContextCard(conv: snapshot?.currentTokens ?? 0,
                            pool: snapshot?.poolTokens ?? 0,
                            total: snapshot?.ctxMax ?? 0)
                    .gridCellColumns(3)
            }
        }
    }

    private var decodeMain: String {
        guard let s = snapshot, s.schedDecode > 0, let v = s.liveDecodeTps else { return "—" }
        return String(format: "%.1f tok/s", v)
    }
    private var prefillMain: String {
        guard let s = snapshot, s.schedPrefill > 0, let v = s.livePrefillTps else { return "—" }
        return String(format: "%.0f tok/s", v)
    }
    private var prefixSub: String {
        var parts: [String] = []
        if let r = snapshot?.cacheReused { parts.append("reusados " + Fmt.tokD(r)) }
        if let u = snapshot?.ssdUsed, let c = snapshot?.ssdCap {
            var t = "SSD " + Fmt.bytes(u) + "/" + Fmt.bytes(c)
            if let dm = snapshot?.ssdDemotions, let rs = snapshot?.ssdRestores, dm + rs > 0 {
                t += " · \(dm)↓ \(rs)↑"
            }
            parts.append(t)
        }
        return parts.joined(separator: " · ")
    }
    private var thermalSub: String {
        let base = [snapshot?.device, snapshot?.ramGB.map { "\($0) GB" }].compactMap { $0 }.joined(separator: " · ")
        let ane = snapshot?.aneShare.map { "ANE \(Int($0 * 100))%" } ?? ""
        return [base, ane].filter { !$0.isEmpty }.joined(separator: " · ")
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

// MARK: - Sparkline (1/s samples, last 2 min)

struct Sparkline: View {
    let values: [Double]
    let color: Color

    var body: some View {
        GeometryReader { geo in
            if values.count > 1 {
                let maxV = Swift.max(values.max() ?? 1, 1)
                Path { p in
                    let w = geo.size.width, h = geo.size.height
                    func pt(_ i: Int, _ v: Double) -> CGPoint {
                        CGPoint(x: w * CGFloat(i) / CGFloat(values.count - 1),
                                y: h * (1 - CGFloat(Swift.min(1, v / maxV))))
                    }
                    p.move(to: pt(0, values[0]))
                    for i in 1..<values.count { p.addLine(to: pt(i, values[i])) }
                }
                .stroke(color, style: StrokeStyle(lineWidth: 1.5, lineJoin: .round))
            }
        }
    }
}

struct SpeedChartCard: View {
    let decode: [Double]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("DECODE · TOK/S")
                    .font(.system(size: 9.5, weight: .semibold))
                    .tracking(0.8)
                    .foregroundStyle(Color.label)
                if let last = decode.last(where: { $0 > 0 }) {
                    Text(String(format: "%.1f", last))
                        .font(.system(size: 9.5).monospacedDigit())
                        .foregroundStyle(Color.sub)
                }
                Spacer()
                Text("últimos 2 min")
                    .font(.system(size: 9.5))
                    .foregroundStyle(Color.label)
            }
            Sparkline(values: decode, color: .accent)
                .frame(height: 26)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.card))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.cardStroke, lineWidth: 1))
    }
}

// MARK: - Finished requests (from the engine log)

struct RecentPanel: View {
    let requests: [RequestDone]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("ÚLTIMAS PETICIONES")
                .font(.system(size: 9.5, weight: .semibold))
                .tracking(0.8)
                .foregroundStyle(Color.label)
            if requests.isEmpty {
                Text("sin peticiones terminadas todavía")
                    .font(.system(size: 11).monospaced())
                    .foregroundStyle(Color.label)
            } else {
                ForEach(requests) { r in
                    let pct = r.input > 0 ? Int(Double(r.cached) / Double(r.input) * 100) : 0
                    Text("\(Fmt.tok(r.input)) → \(Fmt.tok(r.output)) · caché \(pct)% · TTFT \(Fmt.ms(r.ttft * 1000)) · \(String(format: "%.1f", r.tps)) tok/s")
                        .font(.system(size: 11).monospaced())
                        .foregroundStyle(Color.value.opacity(0.85))
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.card))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.cardStroke, lineWidth: 1))
    }
}
