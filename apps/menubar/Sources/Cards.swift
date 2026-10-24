import AppKit
import SwiftUI

extension View {
    func studioCard() -> some View {
        padding(14)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(nsColor: .separatorColor), lineWidth: 0.5))
    }
}

struct StudioNotice: View {
    let message: String
    var isError = false
    let dismiss: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: isError ? "exclamationmark.circle.fill" : "info.circle")
                .foregroundStyle(isError ? Color.red : Color.accentColor)
            Text(message).font(.callout).textSelection(.enabled)
            Spacer(minLength: 0)
            Button(action: dismiss) { Image(systemName: "xmark") }
                .buttonStyle(.borderless).accessibilityLabel("Cerrar mensaje")
        }
        .studioCard()
        .accessibilityElement(children: .contain)
    }
}

private struct StudioMetric: View {
    let title: String
    let value: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(.callout).foregroundStyle(.secondary)
            Text(value).font(.title2.weight(.semibold).monospacedDigit())
                .lineLimit(1).minimumScaleFactor(0.8)
            Text(detail).font(.callout).foregroundStyle(.secondary)
                .lineLimit(2).frame(minHeight: 30, alignment: .topLeading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .studioCard()
        .accessibilityElement(children: .combine)
    }
}

struct StudioSummary: View {
    let snapshot: Snapshot?
    let running: Bool
    let stale: Bool

    // Readiness and freshness are independent: a fresh pressure/failure report
    // still contains useful memory and queue measurements.
    private var available: Snapshot? { stale ? nil : snapshot }

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
            StudioMetric(title: "Generación", value: speed, detail: speedDetail)
            StudioMetric(title: "Memoria Metal", value: available?.memAlloc.map(Fmt.bytes) ?? "—", detail: memoryDetail)
            StudioMetric(title: "Solicitudes activas", value: active, detail: queue)
        }
    }
    private var speed: String {
        guard running, let value = available, value.hasSchedulerMetrics,
              value.schedDecode > 0, let tps = value.liveDecodeTps else { return "—" }
        return String(format: "%.1f tok/s", tps)
    }
    private var speedDetail: String {
        guard let value = available else { return stale ? "Sin actualizar" : "Sin servidor listo" }
        guard running else { return "El motor no está listo" }
        guard value.hasSchedulerMetrics else { return "Estado no disponible" }
        if value.schedPrefill > 0 { return "Preparando el prompt" }
        return value.schedDecode > 0 ? "Velocidad actual" : "En espera"
    }
    private var memoryDetail: String {
        guard let value = available else { return "Sin datos actuales" }
        return [value.memLimit.map { "Límite " + Fmt.bytes($0) }, value.pressure.map { "Presión: " + pressureLabel($0) }]
            .compactMap { $0 }.joined(separator: " · ")
    }
    private func pressureLabel(_ value: String) -> String {
        switch value {
        case "normal": return "normal"
        case "warning": return "alta"
        case "critical": return "crítica"
        default: return value
        }
    }
    private var active: String {
        guard let value = available else { return "—" }
        if value.hasActiveRequests { return String(value.inFlight.count) }
        return value.hasSchedulerMetrics ? String(value.schedDecode + value.schedPrefill) : "—"
    }
    private var queue: String {
        guard let value = available, value.hasSchedulerMetrics else { return "Cola no disponible" }
        return "\(value.schedQueue) en cola"
    }
}

struct StudioActivity: View {
    let snapshot: Snapshot?
    let recent: [RequestDone]
    let stale: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if stale && snapshot != nil {
                Label("Último estado recibido; la actividad puede haber cambiado.", systemImage: "clock")
                    .foregroundStyle(.secondary)
            }
            if let value = snapshot, value.hasActiveRequests {
                if value.inFlight.isEmpty {
                    Text(stale ? "El último estado no tenía solicitudes activas." : "Sin solicitudes activas.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(value.inFlight.prefix(4)) { request in
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Solicitud \(request.id) · \(phase(request.phase))")
                            Text("Prompt: \(Fmt.tok(request.prompt)) · salida: \(Fmt.tok(request.generated)) · \(request.age) s")
                                .foregroundStyle(.secondary).monospacedDigit()
                        }
                    }
                    if value.inFlight.count > 4 {
                        Text("\(value.inFlight.count - 4) solicitudes más.").foregroundStyle(.secondary)
                    }
                }
            } else {
                Text("Actividad actual no disponible.").foregroundStyle(.secondary)
            }
            if !recent.isEmpty {
                Divider()
                Text("Últimas solicitudes").fontWeight(.medium)
                ForEach(recent.prefix(5)) { request in
                    HStack(alignment: .top) {
                        Text("\(Fmt.tok(request.input)) → \(Fmt.tok(request.output)) tokens")
                        Spacer()
                        Text("\(String(format: "%.1f", request.tps)) tok/s · TTFT \(Fmt.ms(request.ttft * 1000))")
                            .foregroundStyle(.secondary)
                    }
                    .monospacedDigit()
                }
            }
        }
        .font(.callout)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    private func phase(_ value: String) -> String {
        switch value {
        case "prefill": return "preparando el prompt"
        case "decode": return "generando"
        case "queued": return "en cola"
        default: return value
        }
    }
}

struct StudioDetails: View {
    let snapshot: Snapshot?
    let decodeSeries: [Double]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !decodeSeries.isEmpty {
                Text("Generación · últimas 120 muestras").font(.callout).foregroundStyle(.secondary)
                StudioSparkline(values: decodeSeries)
                    .frame(height: 42)
                    .accessibilityLabel("Historial de velocidad de generación")
                    .accessibilityValue(decodeSeries.last.map { String(format: "Última muestra: %.1f tokens por segundo", $0) } ?? "Sin muestras")
                Divider()
            }
            detail("Tiempo al primer token", snapshot?.ttftMs.map(Fmt.ms))
            detail("Tiempo entre tokens", snapshot?.itlMs.map(Fmt.ms))
            detail("Aceptación del draft", snapshot?.draftPct.map { String(format: "%.1f%%", $0) })
            detail("KV cache asignada", snapshot?.kvBytes.map(Fmt.bytes))
            detail("Contexto máximo por solicitud", snapshot?.ctxMax.map { Fmt.tok($0) + " tokens" })
            detail("Reutilización de prefijos", snapshot?.cacheHit.map { String(format: "%.1f%%", $0) })
            detail("Caché SSD", ssd)
            detail("Estado de los pesos", snapshot?.weightsState)
            detail("Estado térmico", thermal)
            detail("Neural Engine", ane)
            if let reason = snapshot?.aneReason {
                Text(reason).foregroundStyle(.secondary).textSelection(.enabled)
            }
            if let snapshot, snapshot.hasRequestMetrics {
                detail("Solicitudes terminadas", "\(snapshot.completed) · \(snapshot.cancelled) canceladas · \(snapshot.failed) fallidas")
            }
        }
        .font(.callout)
    }
    private var ssd: String? {
        guard let used = snapshot?.ssdUsed else { return nil }
        return Fmt.bytes(used) + (snapshot?.ssdCap.map { " / " + Fmt.bytes($0) } ?? "")
    }
    private var ane: String? {
        guard let state = snapshot?.aneState else { return nil }
        switch state {
        case "off": return "Desactivado"
        case "stopped": return "Detenido"
        case "split":
            return "Habilitado" + (snapshot?.aneShare.map { String(format: " · FFN configurado al %.0f%%", $0 * 100) } ?? "")
        default: return state
        }
    }
    private var thermal: String? {
        guard let state = snapshot?.thermal else { return nil }
        switch state {
        case "nominal": return "Normal"
        case "fair": return "Moderado"
        case "serious": return "Alto"
        case "critical": return "Crítico"
        default: return state
        }
    }
    private func detail(_ title: String, _ value: String?) -> some View {
        LabeledContent(title) {
            Text(value ?? "—").monospacedDigit().textSelection(.enabled)
        }
    }
}

private struct StudioSparkline: View {
    let values: [Double]

    var body: some View {
        GeometryReader { geometry in
            let maximum = max(values.max() ?? 1, 1)
            if values.count > 1 {
                Path { path in
                    for (index, value) in values.enumerated() {
                        let point = CGPoint(x: geometry.size.width * CGFloat(index) / CGFloat(values.count - 1),
                                            y: geometry.size.height * CGFloat(1 - min(max(value, 0), maximum) / maximum))
                        if index == 0 { path.move(to: point) }
                        else { path.addLine(to: point) }
                    }
                }
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, lineJoin: .round))
            }
        }
    }
}
