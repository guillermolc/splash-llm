import SwiftUI

/// Wraps the persisted Config so SwiftUI can observe/publish settings.
final class ConfigStore: ObservableObject {
    private let config = Config()
    @Published var settings: Settings {
        didSet { config.save() }
    }
    init() { settings = config.settings }
}

// MARK: - Config panel

struct ConfigPanel: View {
    @Binding var s: Settings
    let models: [Engine.Model]
    @Binding var splashRoot: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 10) {
                GridRow {
                    cell("Modelo") {
                        Picker("", selection: $s.model) {
                            ForEach(models) { m in
                                Text("\(m.id)  ·  \(String(format: "%.1f GB", m.sizeGB))")
                                    .tag(m.id)
                            }
                            if !models.contains(where: { $0.id == s.model }) {
                                Text(s.model).tag(s.model)
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: .infinity)
                    }
                    cell("Puerto") {
                        TextField("", text: $s.portText)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 70)
                    }
                }
                GridRow {
                    cell("Contexto") {
                        Picker("", selection: $s.context) {
                            ForEach(["auto", "32K", "64K", "128K", "256K", "512K"], id: \.self) { Text($0).tag($0) }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                    }
                    cell("KV cache") {
                        Picker("", selection: $s.kvFormat) {
                            Text("int8").tag("int8")
                            Text("bf16").tag("bf16")
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                    }
                }
                GridRow {
                    cell("Liberar en idle") {
                        Picker("", selection: $s.idleRelease) {
                            ForEach(["off", "10m", "30m", "1h", "3h", "6h", "8h"], id: \.self) { Text($0).tag($0) }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                    }
                    cell("Mem. máx") {
                        Picker("", selection: $s.maxMemory) {
                            ForEach(["auto", "16G", "24G", "32G", "48G", "64G"], id: \.self) { Text($0).tag($0) }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                    }
                }
                GridRow {
                    cell("SSD cache") {
                        Picker("", selection: $s.cacheDisk) {
                            ForEach(["auto", "0", "8G", "16G", "32G", "64G"], id: \.self) { Text($0).tag($0) }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                    }
                    cell("") {
                        HStack(spacing: 18) {
                            Toggle("Prefill int8", isOn: $s.prefillInt8)
                            Toggle("ANE", isOn: $s.useANE)
                        }
                        .toggleStyle(.checkbox)
                    }
                }
            }
            HStack(spacing: 8) {
                Text("Repo")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.label)
                    .frame(width: 74, alignment: .leading)
                TextField("", text: $splashRoot)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11).monospaced())
            }
        }
    }

    private func cell(_ title: String, @ViewBuilder _ content: () -> some View) -> some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.label)
                .frame(width: 74, alignment: .leading)
            content()
        }
    }
}

extension Settings {
    /// Text binding for the integer port field.
    var portText: String {
        get { String(port) }
        set { port = Int(newValue.filter(\.isNumber)) ?? port }
    }
}

// MARK: - Main window

struct ContentView: View {
    @ObservedObject private var engine = Engine.shared
    @StateObject private var store = ConfigStore()
    @State private var models: [Engine.Model] = []
    @State private var lastError: String?
    @State private var confirmRestart = false
    @State private var startedAt = Date()

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header
            liveSection
            Group {
                Text("CONFIGURACIÓN")
                    .font(.system(size: 9.5, weight: .semibold))
                    .tracking(0.8)
                    .foregroundStyle(Color.label)
                ConfigPanel(s: $store.settings,
                            models: models,
                            splashRoot: $store.settings.splashRoot)
            }
            buttons
            HStack {
                Button("Abrir logs") {
                    NSWorkspace.shared.open(Config.logFileURL())
                }
                .buttonStyle(.plain)
                .font(.system(size: 11))
                .foregroundStyle(Color.label)
                Spacer()
            }
        }
        .padding(20)
        .frame(minWidth: 560, minHeight: 520)
        .background(Color.bg.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onAppear(perform: reloadModels)
        .onChange(of: engine.state) { newValue in
            if newValue == .starting { startedAt = Date() }
            reloadModels()
        }
        .onChange(of: store.settings.splashRoot) { _ in reloadModels() }
        .alert("Splash", isPresented: .init(
                   get: { lastError != nil },
                   set: { if !$0 { lastError = nil } })) {
            Button("Entendido", role: .cancel) {}
        } message: {
            Text(lastError ?? "")
        }
        .confirmationDialog("Reiniciar el engine?",
                            isPresented: $confirmRestart,
                            titleVisibility: .visible) {
            Button("Reiniciar", role: .destructive) { doRestart() }
            Button("Cancelar", role: .cancel) {}
        } message: {
            Text("Se detiene el engine actual y arranca con la configuración elegida.")
        }
    }

    // MARK: Header

    private func uptimeText(_ snap: Snapshot) -> String {
        let base = "\(snap.model) · :\(snap.port)"
        guard let t = snap.startedAt else { return base }
        let s = max(0, Date().timeIntervalSince1970 - t)
        let h = Int(s) / 3600, m = Int(s) / 60 % 60
        return h > 0 ? "\(base) · \(h)h \(m)m" : "\(base) · \(m)m"
    }

    private var header: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(engine.state == .running ? Color.accent : Color.white.opacity(0.15))
                .frame(width: 8, height: 8)
            Text("SPLASH")
                .font(.system(size: 13, weight: .bold))
                .tracking(2)
            if let snap = engine.snapshot, snap.ready || engine.state != .stopped {
                Text(uptimeText(snap))
                    .font(.system(size: 11).monospacedDigit())
                    .foregroundStyle(Color.sub)
            }
            Spacer()
            Text(stateLabel)
                .font(.system(size: 11, weight: .medium).monospacedDigit())
                .foregroundStyle(stateColor)
        }
    }

    private var stateLabel: String {
        switch engine.state {
        case .stopped: return "detenido"
        case .starting: return "cargando…"
        case .running: return engine.snapshot?.inFlight.isEmpty == true
            ? "idle · listo" : "en servicio"
        }
    }
    private var stateColor: Color {
        switch engine.state {
        case .stopped: return Color.label
        case .starting: return Color.warn
        case .running: return Color.accent
        }
    }

    // MARK: Live

    private var liveSection: some View {
        Group {
            if engine.state == .starting {
                loadingPanel
            } else if let snap = engine.snapshot, snap.ready {
                LivePanel(snapshot: snap, decodeSeries: engine.decodeSeries)
                InFlightPanel(requests: snap.inFlight)
                RecentPanel(requests: engine.recent)
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Text("engine detenido")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Color.value)
                    Text("Elegí el modelo y las opciones, y presioná Iniciar. Los números vivos aparecen acá.")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.sub)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 30)
                }
                .padding(12)
                .frame(maxWidth: .infinity, minHeight: 200)
                .background(RoundedRectangle(cornerRadius: 10).fill(Color.card))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.cardStroke, lineWidth: 1))
            }
        }
    }

    private var loadingPanel: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.large)
            Text("cargando \(store.settings.model)…")
                .font(.system(size: 13, weight: .medium).monospacedDigit())
                .foregroundStyle(Color.value)
            if !engine.loadLine.isEmpty {
                Text(engine.loadLine)
                    .font(.system(size: 11).monospaced())
                    .foregroundStyle(Color.sub)
                    .lineLimit(2)
                    .truncationMode(.head)
            }
            TimelineView(.periodic(from: .now, by: 1)) { ctx in
                Text("\(Int(ctx.date.timeIntervalSince(startedAt)))s")
                    .font(.system(size: 11).monospacedDigit())
                    .foregroundColor(Color.sub)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 220)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.card))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.cardStroke, lineWidth: 1))
    }

    // MARK: Buttons

    private var buttons: some View {
        HStack(spacing: 10) {
            Button {
                if engine.state == .stopped { start() }
                else { confirmRestart = engine.snapshot?.inFlight.isEmpty == false }
            } label: {
                Label(engine.state == .stopped ? "Iniciar" : "Reiniciar",
                      systemImage: engine.state == .stopped ? "play.fill" : "arrow.clockwise")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 5)
            }
            .buttonStyle(.borderedProminent)
            .tint(Color.accent)

            Button {
                engine.stop()
            } label: {
                Label("Detener", systemImage: "stop.fill")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 5)
            }
            .buttonStyle(.bordered)
            .disabled(engine.state == .stopped)

            Button {
                engine.releaseMemory(engine.port)
            } label: {
                Label("Liberar memoria", systemImage: "arrow.down.circle")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 5)
            }
            .buttonStyle(.bordered)
            .disabled(engine.state != .running || (engine.snapshot?.weightsIdleMin ?? 0) == 0)
        }
    }

    // MARK: Actions

    private func start() {
        do {
            try engine.start(root: store.settings.splashRoot,
                             model: store.settings.model,
                             s: store.settings)
        } catch {
            lastError = error.localizedDescription
        }
        reloadModels()
    }

    private func doRestart() {
        engine.restart(root: store.settings.splashRoot,
                       model: store.settings.model,
                       s: store.settings)
    }

    private func reloadModels() {
        models = engine.models(root: store.settings.splashRoot)
        if models.isEmpty { models = [Engine.Model(id: store.settings.model, sizeGB: 0)] }
    }
}
