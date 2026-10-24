import AppKit
import SwiftUI

@MainActor
struct ContentView: View {
    @EnvironmentObject private var engine: Engine
    @EnvironmentObject private var store: ConfigStore
    @State private var detailsExpanded = false
    @State private var activityExpanded = false
    @State private var actionError: String?
    @State private var operationInProgress = false
    @State private var confirmation: ConfirmedAction?
    private enum ConfirmedAction { case stop, restart }

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let stale = isStale(at: context.date)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header(stale: stale)
                    if let message = actionError ?? engine.errorMessage ?? store.errorMessage {
                        StudioNotice(message: message, isError: true) {
                            actionError = nil
                            engine.dismissError()
                            store.dismissError()
                        }
                        if engine.state == .failed && engine.ownership == .none {
                            Button("Conectar a un servidor existente") { connect() }
                                .disabled(operationInProgress)
                        }
                    }
                    if engine.state == .starting || engine.state == .stopping {
                        HStack(spacing: 12) {
                            ProgressView().controlSize(.small)
                            Text(engine.loadLine.isEmpty ? stateTitle : engine.loadLine)
                                .font(.callout).foregroundStyle(.secondary)
                                .lineLimit(3).textSelection(.enabled)
                            Spacer(minLength: 0)
                        }
                        .studioCard()
                    }
                    if pendingChanges {
                        HStack(alignment: .top, spacing: 12) {
                            Label("Hay ajustes guardados pendientes de aplicar.", systemImage: "slider.horizontal.3")
                                .font(.callout)
                            Spacer(minLength: 0)
                            Button("Aplicar y reiniciar") { requestRestart() }
                                .disabled(operationInProgress || engine.state != .running)
                        }
                        .studioCard()
                    }
                    if engine.ownership == .external {
                        Label("Servidor externo: Studio solo supervisa esta instancia.", systemImage: "link")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    StudioSummary(snapshot: engine.snapshot, running: engine.state == .running, stale: stale)
                    HStack(spacing: 10) {
                        Button(action: mainAction) {
                            Label(primaryTitle, systemImage: primarySymbol).frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(engine.state == .stopping ||
                                  (operationInProgress && engine.state != .starting))
                        Button {
                            NSWorkspace.shared.open(engine.endpoint)
                        } label: {
                            Label("Abrir chat", systemImage: "bubble.left.and.bubble.right")
                        }
                        .controlSize(.large).disabled(engine.state != .running)
                    }
                    DisclosureGroup("Actividad", isExpanded: $activityExpanded) {
                        StudioActivity(snapshot: engine.snapshot, recent: engine.recent, stale: stale)
                            .padding(.top, 10)
                    }
                    .studioCard()
                    DisclosureGroup("Detalles", isExpanded: $detailsExpanded) {
                        StudioDetails(snapshot: engine.snapshot, decodeSeries: engine.decodeSeries)
                            .padding(.top, 10)
                        HStack {
                            Button("Abrir logs") { NSWorkspace.shared.open(Config.logFileURL()) }
                            Spacer()
                            if engine.ownership == .owned {
                                Button("Liberar memoria") {
                                    perform { try await engine.releaseMemory(engine.port) }
                                }
                                .disabled(!canReleaseMemory || operationInProgress)
                                .help("Libera los pesos en reposo; se cargarán de nuevo al recibir una solicitud.")
                            }
                        }
                        .padding(.top, 10)
                    }
                    .studioCard()
                }
                .padding(20).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(minWidth: 520, minHeight: 420)
        .toolbar {
            if #available(macOS 14, *) {
                SettingsLink { Label("Ajustes", systemImage: "gearshape") }
                    .help("Ajustes de Splash Studio (⌘,)")
            } else {
                Button {
                    NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
                } label: {
                    Label("Ajustes", systemImage: "gearshape")
                }
                .help("Ajustes de Splash Studio (⌘,)")
            }
        }
        .onAppear { engine.setVisible(true) }
        .onDisappear { engine.setVisible(false) }
        .confirmationDialog(confirmation == .restart ? "¿Aplicar los ajustes y reiniciar?" : "¿Detener el servidor?",
                            isPresented: Binding(get: { confirmation != nil },
                                                 set: { if !$0 { confirmation = nil } }),
                            titleVisibility: .visible) {
            if confirmation == .restart {
                Button("Aplicar y reiniciar", role: .destructive) { restart() }
            } else {
                Button("Detener", role: .destructive) { stop() }
            }
            Button("Cancelar", role: .cancel) { confirmation = nil }
        } message: {
            Text("Se interrumpirán las \(activeCount ?? 0) solicitudes activas.")
        }
    }

    private func header(stale: Bool) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(stale ? (engine.state == .failed ? "No disponible · datos anteriores" : "Sin actualizar") : stateTitle,
                      systemImage: stale ? "clock.badge.exclamationmark" : stateSymbol)
                    .font(.headline).foregroundStyle(stale ? Color.secondary : stateColor)
                Spacer()
                if let date = engine.lastUpdated {
                    Text(date, style: .time).font(.callout).foregroundStyle(.secondary)
                        .help("Última respuesta del servidor")
                }
            }
            Text(modelName).font(.title2.weight(.semibold)).lineLimit(2).textSelection(.enabled)
            HStack(spacing: 8) {
                Text(apiURL.absoluteString).font(.callout.monospaced()).foregroundStyle(.secondary)
                    .textSelection(.enabled).lineLimit(1).truncationMode(.middle)
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(apiURL.absoluteString, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless).help("Copiar URL de la API")
                .accessibilityLabel("Copiar URL de la API")
                Spacer(minLength: 0)
            }
        }
    }

    private var modelName: String {
        if engine.state == .stopped || engine.ownership == .none { return store.settings.model }
        if engine.state == .running || engine.state == .starting,
           let model = engine.snapshot?.model, model != "—" { return model }
        return engine.appliedSettings?.model ?? store.settings.model
    }
    private var apiURL: URL {
        let root = engine.ownership == .none && engine.state != .starting
            ? (URL(string: "http://127.0.0.1:\(store.settings.port)") ?? engine.endpoint) : engine.endpoint
        return root.appendingPathComponent("v1")
    }
    private var activeCount: Int? {
        guard let snapshot = engine.snapshot else { return nil }
        if snapshot.hasActiveRequests { return snapshot.inFlight.count }
        if snapshot.hasSchedulerMetrics { return snapshot.schedDecode + snapshot.schedPrefill }
        return nil
    }
    private var pendingChanges: Bool {
        engine.ownership == .owned && engine.appliedSettings.map { $0 != store.settings } == true
    }
    private var canReleaseMemory: Bool {
        engine.state == .running && !isStale(at: Date()) && activeCount == 0 &&
        engine.snapshot?.ready == true && engine.snapshot?.weightsState == "cargados"
    }
    private func isStale(at date: Date) -> Bool {
        guard engine.snapshot != nil else { return false }
        guard engine.ownership != .none else { return false }
        guard let last = engine.lastUpdated else { return true }
        return date.timeIntervalSince(last) > 10
    }
    private var stateTitle: String {
        switch engine.state {
        case .stopped: return "Detenido"
        case .starting: return "Iniciando"
        case .stopping: return "Deteniendo"
        case .failed: return "No disponible"
        case .running:
            if engine.snapshot?.weightsState == "restaurando" { return "Restaurando pesos" }
            if let count = activeCount, count > 0 { return "En servicio" }
            return engine.snapshot?.ready == true ? "Listo" : "Preparando"
        }
    }
    private var stateSymbol: String {
        switch engine.state {
        case .running: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.circle.fill"
        case .starting, .stopping: return "arrow.triangle.2.circlepath"
        case .stopped: return "stop.circle"
        }
    }
    private var stateColor: Color {
        switch engine.state {
        case .running: return .accentColor
        case .failed: return .red
        case .starting, .stopping: return .orange
        case .stopped: return .secondary
        }
    }
    private var primaryTitle: String {
        switch engine.state {
        case .running: return engine.ownership == .external ? "Desconectar" : "Detener"
        case .starting: return "Cancelar inicio"
        case .stopping: return "Deteniendo…"
        case .failed:
            if engine.ownership == .owned { return "Reintentar detener" }
            return engine.ownership == .external ? "Reconectar" : "Reintentar"
        case .stopped: return "Iniciar"
        }
    }
    private var primarySymbol: String {
        switch engine.state {
        case .running, .starting, .stopping: return "stop.fill"
        case .failed: return engine.ownership == .owned ? "stop.fill" : "arrow.clockwise"
        case .stopped: return "play.fill"
        }
    }
    private func mainAction() {
        switch engine.state {
        case .running:
            if engine.ownership == .owned && (activeCount ?? 0) > 0 { confirmation = .stop }
            else { stop() }
        case .starting: stop()
        case .failed where engine.ownership == .owned: stop()
        case .failed where engine.ownership == .external:
            let port = engine.port
            perform {
                await engine.stop()
                await engine.adoptIfRunning(port: port)
            }
        case .stopped, .failed:
            perform {
                let settings = store.settings
                try await engine.start(root: settings.splashRoot, model: settings.model, s: settings)
            }
        case .stopping: break
        }
    }
    private func stop() {
        confirmation = nil
        Task { await engine.stop() }
    }
    private func connect() {
        perform { await engine.adoptIfRunning(port: store.settings.port) }
    }
    private func requestRestart() {
        if (activeCount ?? 0) > 0 { confirmation = .restart }
        else { restart() }
    }
    private func restart() {
        confirmation = nil
        perform {
            let settings = store.settings
            try await engine.restart(root: settings.splashRoot, model: settings.model, s: settings)
        }
    }
    private func perform(_ operation: @escaping @MainActor () async throws -> Void) {
        guard !operationInProgress else { return }
        operationInProgress = true
        actionError = nil
        Task {
            defer { operationInProgress = false }
            do { try await operation() }
            catch is CancellationError { }
            catch { actionError = error.localizedDescription }
        }
    }
}

@MainActor
struct StudioSettingsView: View {
    @EnvironmentObject private var engine: Engine
    @EnvironmentObject private var store: ConfigStore
    @State private var draft = Settings.default
    @State private var portText = ""
    @State private var models: [Engine.Model] = []
    @State private var advanced = false
    @State private var loadingModels = false
    @State private var saveError: String?
    @State private var saved = false

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("Servidor") {
                    Picker("Modelo", selection: $draft.model) {
                        ForEach(models) { model in Text(model.id).tag(model.id) }
                        if !models.contains(where: { $0.id == draft.model }) {
                            Text(draft.model.isEmpty ? "Selecciona un modelo" : draft.model).tag(draft.model)
                        }
                    }
                    HStack {
                        Text(loadingModels ? "Buscando modelos instalados…" : "Solo se usan modelos ya instalados.")
                            .font(.callout).foregroundStyle(.secondary)
                        Spacer()
                        Button("Actualizar") { Task { await loadModels(force: true) } }
                            .disabled(loadingModels)
                    }
                    Picker("Contexto máximo", selection: $draft.context) {
                        ForEach(contextOptions, id: \.self) { value in
                            Text(value == "auto" ? "Automático" : value).tag(value)
                                .disabled(!isSupportedContext(value))
                        }
                    }
                    Text(contextLimitDescription)
                        .font(.callout).foregroundStyle(.secondary)
                    Picker("Límite de memoria", selection: $draft.maxMemory) {
                        ForEach(memoryOptions, id: \.self) { value in
                            Text(value == "auto" ? "Automático" : value).tag(value)
                        }
                    }
                }
                Section {
                    DisclosureGroup("Avanzado", isExpanded: $advanced) {
                        TextField("Puerto", text: $portText)
                            .onChange(of: portText) { draft.port = Int($0) ?? 0 }
                        Picker("Formato de KV cache", selection: $draft.kvFormat) {
                            Text("INT8").tag("int8")
                            Text("BF16").tag("bf16")
                        }
                        Toggle("Usar Neural Engine", isOn: $draft.useANE)
                        Toggle("Prefill con activaciones INT8", isOn: $draft.prefillInt8)
                            .help("Puede acelerar la preparación del prompt y cambiar resultados numéricos.")
                        Picker("Caché en SSD", selection: $draft.cacheDisk) {
                            ForEach(cacheOptions, id: \.self) { value in
                                Text(value == "auto" ? "Automática" : value == "0" ? "Desactivada" : value).tag(value)
                            }
                        }
                        Toggle("Conservar caché entre reinicios", isOn: $draft.persistentCache)
                            .disabled(draft.cacheDisk == "0")
                        Picker("Liberar pesos por inactividad", selection: $draft.idleRelease) {
                            ForEach(idleOptions, id: \.self) { value in
                                Text(value == "off" ? "Nunca" : value).tag(value)
                            }
                        }
                        HStack {
                            TextField("Carpeta de Splash", text: $draft.splashRoot)
                            Button("Elegir…", action: chooseRoot)
                        }
                    }
                }
                if !validationErrors.isEmpty {
                    Section("Revisa estos ajustes") {
                        ForEach(validationErrors, id: \.self) { error in
                            Label(error, systemImage: "exclamationmark.circle")
                                .foregroundStyle(.red).font(.callout)
                        }
                    }
                }
                if let message = saveError ?? store.errorMessage {
                    Section { Text(message).foregroundStyle(.red).textSelection(.enabled) }
                }
            }
            .formStyle(.grouped)
            Divider()
            HStack(spacing: 12) {
                Text(saved ? "Ajustes guardados." : "Los cambios se aplican al iniciar o reiniciar el servidor.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button("Revertir") { resetDraft() }
                    .fixedSize().disabled(draft == store.settings)
                Button("Guardar", action: save)
                    .fixedSize()
                    .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
                    .disabled(draft == store.settings || !validationErrors.isEmpty)
            }
            .padding(16)
        }
        .frame(width: 600)
        .frame(minHeight: 460, idealHeight: 560)
        .onAppear(perform: resetDraft)
        .onChange(of: draft) { value in if value != store.settings { saved = false } }
        .task(id: draft.splashRoot) {
            do { try await Task.sleep(nanoseconds: 300_000_000) }
            catch { return }
            await loadModels(force: false)
        }
    }

    private var contextLimit: Int {
        min(models.first(where: { $0.id == draft.model })?.maximumContextTokens ?? 262_144, 262_144)
    }
    private var contextLimitDescription: String {
        guard let limit = models.first(where: { $0.id == draft.model })?.maximumContextTokens else {
            return "El motor validará el límite de contexto del modelo."
        }
        return "Máximo permitido: \(Fmt.tok(min(limit, 262_144))) tokens."
    }
    private var contextOptions: [String] {
        preservingSelection(["auto", "4K", "8K", "16K", "32K", "64K", "128K", "256K"].filter(isSupportedContext), draft.context)
    }
    private var memoryOptions: [String] {
        preservingSelection(["auto", "16G", "24G", "32G", "48G", "64G", "96G"], draft.maxMemory)
    }
    private var cacheOptions: [String] {
        preservingSelection(["auto", "0", "8G", "16G", "32G", "64G"], draft.cacheDisk)
    }
    private var idleOptions: [String] {
        preservingSelection(["off", "10m", "30m", "1h", "3h", "6h", "8h"], draft.idleRelease)
    }
    private var validationErrors: [String] { draft.validationErrors(maximumContextTokens: contextLimit) }
    private func preservingSelection(_ options: [String], _ value: String) -> [String] {
        options.contains(value) ? options : options + [value]
    }
    private func isSupportedContext(_ value: String) -> Bool {
        if value == "auto" { return true }
        var candidate = draft
        candidate.context = value
        guard let tokens = candidate.contextTokens else { return false }
        return tokens > 0 && tokens <= contextLimit
    }
    private func resetDraft() {
        draft = store.settings
        portText = String(draft.port)
        saveError = nil
        saved = false
    }
    private func save() {
        do {
            try store.save(draft)
            draft = store.settings
            portText = String(draft.port)
            saveError = nil
            saved = true
        } catch { saveError = error.localizedDescription }
    }
    private func loadModels(force: Bool) async {
        let root = draft.splashRoot
        guard !root.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            models = []
            loadingModels = false
            return
        }
        loadingModels = true
        let found = await engine.models(root: root, force: force)
        guard !Task.isCancelled, root == draft.splashRoot else { return }
        models = found
        loadingModels = false
    }
    private func chooseRoot() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Elegir"
        panel.directoryURL = URL(fileURLWithPath: draft.splashRoot)
        panel.begin { result in
            if result == .OK, let url = panel.url { draft.splashRoot = url.path }
        }
    }
}
