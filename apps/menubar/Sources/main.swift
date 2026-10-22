import AppKit
import Foundation

// MARK: - CLI one-shots (also used to smoke-test the HTTP paths without a GUI)

func runCLI(_ args: [String]) {
    let cfg = Config()
    let root = cfg.settings.splashRoot
    switch args.first {
    case "--models":
        for m in Engine().models(root: root) {
            print(String(format: "%-@  %4.1f GB", m.id, m.sizeGB))
        }
    case "--status":
        let port = args.count > 1 ? (Int(args[1]) ?? cfg.settings.port) : cfg.settings.port
        guard let s = Engine().statusSync(port) else {
            print("sin respuesta en :\(port)")
            exit(1)
        }
        print(StatusPanel.format(s))
    case "--release":
        let port = args.count > 1 ? (Int(args[1]) ?? cfg.settings.port) : cfg.settings.port
        Engine().releaseMemory(port)
        print("pedido de liberación enviado a :\(port)")
    case "--logs":
        NSWorkspace.shared.open(Config.logFileURL().deletingLastPathComponent())
    default:
        print("usos: --models | --status [puerto] | --release [puerto] | --logs")
        exit(1)
    }
}

// MARK: - Status formatting

enum StatusPanel {
    static func format(_ s: [String: Any]) -> String {
        var lines: [String] = []
        if let instance = s["instance"] as? [String: Any] {
            let model = instance["model"] as? String ?? "?"
            let port = instance["port"] as? Int ?? 0
            lines.append("modelo   \(model)")
            lines.append("puerto   :\(port)")
        }
        if let ctx = s["maximum_context_tokens"] as? Int {
            lines.append("ctx max  \(tok(ctx))")
        }
        if let kv = s["kv"] as? [String: Any] {
            var parts: [String] = []
            if let b = kv["allocated_bytes"] as? Double { parts.append(bytes(b)) }
            if let active = kv["pages_active"] as? Int { parts.append("\(active) pág activas") }
            if let free = kv["pages_free"] as? Int { parts.append("\(free) libres") }
            if !parts.isEmpty { lines.append("KV       " + parts.joined(separator: " · ")) }
        }
        if let gov = s["memory_governor"] as? [String: Any] {
            let charged = gov["charged_bytes"] as? Double ?? 0
            let limit = gov["limit_bytes"] as? Double ?? 0
            var text = bytes(charged)
            if limit > 0 { text += " / " + bytes(limit) }
            lines.append("mem      " + text)
        }
        if let pressure = s["memory_pressure"] as? String {
            lines.append("presión  " + pressure)
        }
        if let timing = s["model_timing"] as? [String: Any],
           let tps = timing["decode_tokens_per_second"] as? Double, tps > 0 {
            lines.append("decode   \(Int(tps)) tok/s")
        }
        if let requests = s["requests"] as? [[String: Any]], !requests.isEmpty {
            lines.append("peticiones activas  " + "\(requests.count)")
        }
        if let weights = s["weights"] as? [String: Any], let state = weights["state"] as? String {
            lines.append("pesos    " + state)
        }
        return lines.joined(separator: "\n")
    }

    static func tok(_ n: Int) -> String {
        n >= 1_000_000 ? String(format: "%.1fM", Double(n) / 1_000_000)
            : n >= 1_000 ? String(format: "%.1fk", Double(n) / 1_000)
            : "\(n)"
    }

    static func bytes(_ n: Double) -> String {
        if n >= 1_000_000_000 { return String(format: "%.1f GB", n / 1_000_000_000) }
        if n >= 1_000_000 { return String(format: "%.0f MB", n / 1_000_000) }
        return String(format: "%.0f KB", n / 1_000)
    }
}

// MARK: - Menu plumbing

final class ActionItem: NSMenuItem {
    var closure: (() -> Void)?
    @objc func fire() { closure?() }
    init(_ title: String, _ action: @escaping () -> Void) {
        super.init(title: title, action: #selector(fire), keyEquivalent: "")
        self.closure = action
        self.target = self
    }
    required init(coder: NSCoder) { fatalError("unused") }
}

func disabledItem(_ attributed: NSAttributedString) -> NSMenuItem {
    let item = NSMenuItem()
    item.attributedTitle = attributed
    item.isEnabled = false
    return item
}

func coloredLine(_ text: String, _ color: NSColor = .labelColor) -> NSAttributedString {
    NSAttributedString(string: text, attributes: [
        .foregroundColor: color,
        .font: NSFont.monospacedDigitSystemFont(ofSize: NSFont.systemFontSize, weight: .regular),
    ])
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let config = Config()
    let engine = Engine()
    var statusItem: NSStatusItem?
    var menu: NSMenu?
    var menuIsOpen = false
    var refreshTimer: Timer?
    var didLogFirstUpdate = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = NSImage(systemSymbolName: "brain", accessibilityDescription: "Splash")
        let m = NSMenu()
        m.delegate = self
        m.autoenablesItems = false
        item.menu = m
        statusItem = item
        menu = m
        engine.port = config.settings.port
        engine.onChange = { [weak self] in self?.refreshTint() }
        engine.startPolling()
        refreshTint()
        // Populate immediately so the menu is never empty, and keep it
        // fresh while the app lives (independent of delegate callbacks).
        rebuild()
        FileHandle.standardError.write(Data(
            "splash-menubar: listo (menú con \(menu?.numberOfItems ?? 0) items)\n".utf8))
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self, !self.menuIsOpen else { return }
            self.rebuild()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        engine.stop()
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        menuIsOpen = true
        rebuild()
        if !didLogFirstUpdate {
            didLogFirstUpdate = true
            FileHandle.standardError.write(Data(
                "splash-menubar: menuNeedsUpdate → \(menu.numberOfItems) items\n".utf8))
        }
    }

    func menuWillOpen(_ menu: NSMenu) {
        menuIsOpen = true
    }

    func menuDidClose(_ menu: NSMenu) {
        menuIsOpen = false
    }

    private func refreshTint() {
        let color: NSColor
        switch engine.state {
        case .stopped: color = .systemGray
        case .starting: color = .systemYellow
        case .running:
            color = (engine.lastStatus?["memory_pressure"] as? String) == "critical"
                ? .systemRed : .systemGreen
        }
        statusItem?.button?.contentTintColor = color
    }

    // MARK: Menu construction

    private func rebuild() {
        // Build into the standing menu; NSMenu refuses to move items that
        // already belong to a different menu, so a scratch menu would end
        // up displaying empty.
        if menu == nil {
            let m0 = NSMenu()
            m0.delegate = self
            m0.autoenablesItems = false
            menu = m0
            statusItem?.menu = m0
        }
        guard let m = menu else { return }
        m.removeAllItems()
        m.autoenablesItems = false
        let s = config.settings

        // Header.
        let header = NSMenuItem()
        let stateText: String
        let stateColor: NSColor
        switch engine.state {
        case .stopped: (stateText, stateColor) = ("detenido", .systemGray)
        case .starting: (stateText, stateColor) = ("cargando…", .systemYellow)
        case .running: (stateText, stateColor) = ("activo", .systemGreen)
        }
        let model = (engine.lastStatus?["instance"] as? [String: Any])?["model"] as? String
            ?? s.model
        header.attributedTitle = NSAttributedString(
            string: "Splash  ·  \(stateText)  ·  \(model)",
            attributes: [.foregroundColor: stateColor,
                          .font: NSFont.boldSystemFont(ofSize: NSFont.systemFontSize)])
        header.isEnabled = false
        m.addItem(header)
        m.addItem(.separator())

        // Model.
        let modelItem = NSMenuItem(title: "Modelo", action: nil, keyEquivalent: "")
        let modelMenu = NSMenu()
        let models = engine.models(root: s.splashRoot)
        if models.isEmpty {
            let none = NSMenuItem(title: "(ninguno instalado)", action: nil, keyEquivalent: "")
            none.isEnabled = false
            modelMenu.addItem(none)
        }
        for m2 in models {
            let it = ActionItem(String(format: "%@  (%.1f GB)", m2.id, m2.sizeGB)) { [weak self] in
                guard let self else { return }
                self.config.settings.model = m2.id
                self.config.save()
                self.restartIfNeeded()
                self.rebuild()
            }
            it.state = m2.id == s.model ? .on : .off
            modelMenu.addItem(it)
        }
        modelItem.submenu = modelMenu
        m.addItem(modelItem)

        // Context.
        let ctxItem = NSMenuItem(title: "Contexto", action: nil, keyEquivalent: "")
        let ctxMenu = NSMenu()
        for value in ["16K", "32K", "64K", "128K", "auto"] {
            let it = ActionItem(value) { [weak self] in
                guard let self else { return }
                self.config.settings.context = value
                self.config.save()
                self.restartIfNeeded()
                self.rebuild()
            }
            it.state = value == s.context ? .on : .off
            ctxMenu.addItem(it)
        }
        ctxItem.submenu = ctxMenu
        m.addItem(ctxItem)

        // KV format.
        let kvItem = NSMenuItem(title: "KV", action: nil, keyEquivalent: "")
        let kvMenu = NSMenu()
        for value in ["int8", "bf16"] {
            let it = ActionItem(value) { [weak self] in
                guard let self else { return }
                self.config.settings.kvFormat = value
                self.config.save()
                self.restartIfNeeded()
                self.rebuild()
            }
            it.state = value == s.kvFormat ? .on : .off
            kvMenu.addItem(it)
        }
        kvItem.submenu = kvMenu
        m.addItem(kvItem)

        // Idle release.
        let idleItem = NSMenuItem(title: "Liberar inactivo", action: nil, keyEquivalent: "")
        let idleMenu = NSMenu()
        for value in ["off", "10m", "30m", "2h"] {
            let it = ActionItem(value) { [weak self] in
                guard let self else { return }
                self.config.settings.idleRelease = value
                self.config.save()
                self.restartIfNeeded()
                self.rebuild()
            }
            it.state = value == s.idleRelease ? .on : .off
            idleMenu.addItem(it)
        }
        idleItem.submenu = idleMenu
        m.addItem(idleItem)

        // Max memory.
        let memItem = NSMenuItem(title: "Techo de memoria", action: nil, keyEquivalent: "")
        let memMenu = NSMenu()
        for value in ["auto", "24G", "28G", "32G"] {
            let it = ActionItem(value) { [weak self] in
                guard let self else { return }
                self.config.settings.maxMemory = value
                self.config.save()
                self.restartIfNeeded()
                self.rebuild()
            }
            it.state = value == s.maxMemory ? .on : .off
            memMenu.addItem(it)
        }
        memItem.submenu = memMenu
        m.addItem(memItem)

        // SSD cache.
        let cacheItem = NSMenuItem(title: "Caché SSD", action: nil, keyEquivalent: "")
        let cacheMenu = NSMenu()
        for value in ["auto", "16G", "0"] {
            let label = value == "0" ? "off" : value
            let it = ActionItem(label) { [weak self] in
                guard let self else { return }
                self.config.settings.cacheDisk = value
                self.config.save()
                self.restartIfNeeded()
                self.rebuild()
            }
            it.state = value == s.cacheDisk ? .on : .off
            cacheMenu.addItem(it)
        }
        cacheItem.submenu = cacheMenu
        m.addItem(cacheItem)

        // Toggles.
        let ane = ActionItem("Neural Engine (ANE)") { [weak self] in
            guard let self else { return }
            self.config.settings.useANE.toggle()
            self.config.save()
            self.restartIfNeeded()
            self.rebuild()
        }
        ane.state = s.useANE ? .on : .off
        m.addItem(ane)
        let int8 = ActionItem("Prefill int8 (PR #445)") { [weak self] in
            guard let self else { return }
            self.config.settings.prefillInt8.toggle()
            self.config.save()
            self.restartIfNeeded()
            self.rebuild()
        }
        int8.state = s.prefillInt8 ? .on : .off
        m.addItem(int8)

        m.addItem(.separator())

        // Live status block.
        if engine.state == .running, let status = engine.lastStatus {
            let pressure = status["memory_pressure"] as? String
            let dotColor: NSColor = pressure == "critical" ? .systemRed
                : pressure == "warning" ? .systemOrange : .systemGreen
            let kv = status["kv"] as? [String: Any]
            let kvBytes = kv?["allocated_bytes"] as? Double ?? 0
            let gov = status["memory_governor"] as? [String: Any]
            let charged = gov?["charged_bytes"] as? Double ?? 0
            let limit = gov?["limit_bytes"] as? Double ?? 0
            let planModel = (status["memory_plan"] as? [String: Any])?["model"] as? [String: Any]
            let ctxMax = status["maximum_context_tokens"] as? Int
                ?? planModel?["maximum_context_tokens"] as? Int
            var line = "●  mem " + StatusPanel.bytes(charged)
            if limit > 0 { line += " / " + StatusPanel.bytes(limit) }
            if kvBytes > 0 { line += "   ·   KV " + StatusPanel.bytes(kvBytes) }
            if let ctxMax { line += "   ·   ctx " + StatusPanel.tok(ctxMax) }
            if let timing = status["model_timing"] as? [String: Any],
               let tps = timing["decode_tokens_per_second"] as? Double, tps > 0 {
                line += "   ·   " + "\(Int(tps)) tok/s"
            }
            m.addItem(disabledItem(coloredLine(line, dotColor)))
        } else if engine.state == .starting {
            m.addItem(disabledItem(coloredLine("●  cargando modelo…", .systemYellow)))
        } else {
            m.addItem(disabledItem(coloredLine("●  engine detenido", .systemGray)))
        }

        m.addItem(.separator())

        // Actions.
        let toggle = ActionItem(engine.state == .stopped ? "▶  Iniciar" : "⏹  Detener") { [weak self] in
            guard let self else { return }
            if self.engine.state == .stopped {
                self.startEngine()
            } else {
                self.engine.stop()
            }
            self.rebuild()
        }
        m.addItem(toggle)

        if engine.state == .running {
            let release = ActionItem("Liberar memoria (pesos)") { [weak self] in
                guard let self else { return }
                self.engine.releaseMemory(self.config.settings.port)
            }
            m.addItem(release)
        }
        let logs = ActionItem("Abrir logs") {
            NSWorkspace.shared.open(Config.logFileURL())
        }
        m.addItem(logs)
        let settings = ActionItem("Ajustes…") { [weak self] in
            self?.editRoot()
        }
        m.addItem(settings)
        m.addItem(.separator())
        let quit = ActionItem("Salir") {
            NSApp.terminate(nil)
        }
        m.addItem(quit)
    }

    private func startEngine() {
        let s = config.settings
        engine.port = s.port
        do {
            try engine.start(root: s.splashRoot, model: s.model, s: s)
        } catch {
            let alert = NSAlert()
            alert.messageText = "No pudo iniciar"
            alert.informativeText = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            alert.runModal()
        }
    }

    private func restartIfNeeded() {
        guard engine.state != .stopped else { return }
        let alert = NSAlert()
        alert.messageText = "Reiniciar el engine?"
        alert.informativeText = "Los cambios se aplican al reiniciar. Las peticiones en curso se interrumpen."
        alert.addButton(withTitle: "Reiniciar")
        alert.addButton(withTitle: "Cancelar")
        guard alert.runModal() == .alertFirstButtonReturn else {
            // Revert the change we just saved.
            rebuild()
            return
        }
        engine.stop()
        startEngine()
    }

    private func editRoot() {
        let alert = NSAlert()
        alert.messageText = "Carpeta del proyecto splash-llm"
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 380, height: 24))
        field.stringValue = config.settings.splashRoot
        alert.accessoryView = field
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancelar")
        NSApp.activate(ignoringOtherApps: true)
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let path = field.stringValue
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            return
        }
        config.settings.splashRoot = path
        config.save()
        rebuild()
    }
}

// MARK: - Entry point

let arguments = Array(CommandLine.arguments.dropFirst())
if !arguments.isEmpty {
    runCLI(Array(arguments))
    exit(0)
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
