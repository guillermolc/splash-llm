import AppKit
import Foundation

// MARK: - Formatting

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

enum StatusPanel {
    static func format(_ s: [String: Any]) -> String {
        var lines: [String] = []
        if let instance = s["instance"] as? [String: Any] {
            lines.append("modelo   \(instance["model"] as? String ?? "?")")
            lines.append("puerto   :\(instance["port"] as? Int ?? 0)")
        }
        if let m = s["metrics"] as? [String: Any] {
            if let tps = m["decode_tokens_per_second"] as? Double, tps > 0 {
                lines.append("decode   \(Int(tps)) tok/s")
            }
            if let tps = m["prefill_tokens_per_second"] as? Double, tps > 0 {
                lines.append("prefill  \(Int(tps)) tok/s")
            }
        }
        if let kv = s["kv"] as? [String: Any] {
            var parts: [String] = []
            if let b = kv["allocated_bytes"] as? Double { parts.append(Fmt.bytes(b)) }
            if let a = kv["pages_active"] as? Int, let total = kv["pages_allocated"] as? Int {
                parts.append("\(a)/\(total) pág")
            }
            if !parts.isEmpty { lines.append("KV       " + parts.joined(separator: " · ")) }
        }
        if let mem = s["memory_actual"] as? [String: Any], let a = mem["allocated_bytes"] as? Double {
            lines.append("mem      \(Fmt.bytes(a))")
        }
        if let pressure = s["memory_pressure"] as? String { lines.append("presión  " + pressure) }
        if let requests = s["requests"] as? [String: Any] {
            lines.append("peticiones  \(requests["submitted"] as? Int ?? 0) enviadas · \(requests["completed"] as? Int ?? 0) hechas")
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Window UI plumbing

func labelRow(_ label: String) -> (view: NSView, value: NSTextField) {
    let l = NSTextField(labelWithString: label)
    l.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
    l.textColor = .secondaryLabelColor
    l.frame.size.width = 110
    let v = NSTextField(labelWithString: "—")
    v.font = .monospacedDigitSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular)
    let stack = NSStackView(views: [l, v])
    stack.orientation = .horizontal
    stack.spacing = 8
    stack.setHuggingPriority(.defaultLow, for: .horizontal)
    return (stack, v)
}

/// A menu item that fires a closure instead of a selector.
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

// MARK: - App

final class AppController: NSObject, NSApplicationDelegate, NSWindowDelegate {
    let config = Config()
    let engine = Engine()

    var window: NSWindow?
    var statusItem: NSStatusItem?

    // Config controls.
    var modelPopup: NSPopUpButton?
    var portField: NSTextField?
    var ctxSegment: NSSegmentedControl?
    var kvSegment: NSSegmentedControl?
    var idleSegment: NSSegmentedControl?
    var memSegment: NSSegmentedControl?
    var cacheSegment: NSSegmentedControl?
    var aneCheck: NSButton?
    var int8Check: NSButton?

    // Live value fields.
    var live: [String: NSTextField] = [:]
    var headerValue: NSTextField?
    var apiLabel: NSTextField?
    var inFlightStack: NSStackView?
    var inFlightKey: String = ""

    var startButton: NSButton?
    var stopButton: NSButton?
    var releaseButton: NSButton?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildMainMenu()
        buildWindow()
        buildStatusItem()
        engine.port = config.settings.port
        engine.onChange = { [weak self] in self?.tick() }
        engine.adoptIfRunning()
        engine.startPolling()
        reloadModels()
        syncControls()
        tick()
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationWillTerminate(_ notification: Notification) {
        engine.stop()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        window?.makeKeyAndOrderFront(nil)
        return true
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)   // hide; the app (and the engine) keep running
        return false
    }

    // MARK: Menus

    private func buildMainMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        let quit = NSMenuItem(title: "Salir", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appMenu.addItem(quit)
        appItem.submenu = appMenu
        main.addItem(appItem)
        NSApp.mainMenu = main
    }

    private func buildStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = NSImage(systemSymbolName: "brain", accessibilityDescription: "Splash")
        let menu = NSMenu()
        let open = ActionItem("Mostrar ventana") { [weak self] in
            self?.window?.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
        let stop = ActionItem("Detener engine") { [weak self] in self?.engine.stop() }
        let quit = ActionItem("Salir") { NSApp.terminate(nil) }
        menu.addItem(open)
        menu.addItem(stop)
        menu.addItem(quit)
        item.menu = menu
        statusItem = item
    }

    // MARK: Window

    private func buildWindow() {
        let w = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 640, height: 720),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        w.title = "Splash"
        w.isReleasedWhenClosed = false
        w.delegate = self
        w.minSize = NSSize(width: 520, height: 480)

        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 12
        root.edgeInsets = NSEdgeInsets(top: 16, left: 16, bottom: 16, right: 16)
        root.translatesAutoresizingMaskIntoConstraints = false

        // Header.
        let dot = NSTextField(labelWithString: "●")
        dot.font = .systemFont(ofSize: 15)
        let title = NSTextField(labelWithString: "Splash")
        title.font = .boldSystemFont(ofSize: 15)
        headerValue = {
            let v = NSTextField(labelWithString: "")
            v.font = .systemFont(ofSize: 12)
            v.textColor = .secondaryLabelColor
            return v
        }()
        let header = NSStackView(views: [dot, title, headerValue!])
        header.spacing = 8
        let apiLabel = NSTextField(labelWithString: "API  http://127.0.0.1:\(config.settings.port)/v1")
        apiLabel.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        apiLabel.textColor = .secondaryLabelColor
        self.apiLabel = apiLabel
        let copyBtn = NSButton(title: "copiar", target: self, action: #selector(copyAPI))
        copyBtn.bezelStyle = .rounded
        copyBtn.controlSize = .small
        let apiRow = NSStackView(views: [apiLabel, copyBtn])
        apiRow.spacing = 6
        let headerRow = NSStackView(views: [header, apiRow])
        headerRow.orientation = .horizontal
        headerRow.distribution = .fill
        headerRow.alignment = .firstBaseline
        root.addArrangedSubview(headerRow)

        // Configuration.
        let cfgBox = NSBox()
        cfgBox.title = "Modelo y ajustes"
        cfgBox.titlePosition = .atTop
        let cfg = NSStackView()
        cfg.orientation = .vertical
        cfg.spacing = 10
        cfg.translatesAutoresizingMaskIntoConstraints = true

        cfg.addArrangedSubview(controlRow("Modelo", {
            let p = NSPopUpButton()
            p.target = self
            p.action = #selector(modelChanged)
            self.modelPopup = p
            return p
        }))
        cfg.addArrangedSubview(controlRow("Puerto", {
            let f = NSTextField(string: "\(config.settings.port)")
            f.widthAnchor.constraint(equalToConstant: 70).isActive = true
            f.target = self
            f.action = #selector(portChanged)
            f.translatesAutoresizingMaskIntoConstraints = false
            self.portField = f
            return f
        }))
        cfg.addArrangedSubview(controlRow("Contexto", {
            let s = NSSegmentedControl(labels: ["16K", "32K", "64K", "128K", "auto"],
                                       trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
            self.ctxSegment = s
            return s
        }))
        let kvRow = controlRow("KV", {
            let s = NSSegmentedControl(labels: ["int8", "bf16"],
                                       trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
            self.kvSegment = s
            return s
        })
        let idleRow = controlRow("Liberar inactivo", {
            let s = NSSegmentedControl(labels: ["off", "10m", "30m", "2h"],
                                       trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
            self.idleSegment = s
            return s
        })
        let kvIdle = NSStackView(views: [kvRow, idleRow])
        kvIdle.distribution = .fillEqually
        kvIdle.spacing = 12
        cfg.addArrangedSubview(kvIdle)
        let memRow = controlRow("Techo de memoria", {
            let s = NSSegmentedControl(labels: ["auto", "24G", "28G", "32G"],
                                       trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
            self.memSegment = s
            return s
        })
        let cacheRow = controlRow("Caché SSD", {
            let s = NSSegmentedControl(labels: ["auto", "16G", "off"],
                                       trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
            self.cacheSegment = s
            return s
        })
        let memCache = NSStackView(views: [memRow, cacheRow])
        memCache.distribution = .fillEqually
        memCache.spacing = 12
        cfg.addArrangedSubview(memCache)
        let ane = NSButton(checkboxWithTitle: "Neural Engine (ANE)", target: self, action: #selector(settingsChanged))
        ane.state = config.settings.useANE ? .on : .off
        aneCheck = ane
        let int8 = NSButton(checkboxWithTitle: "Prefill int8", target: self, action: #selector(settingsChanged))
        int8.state = config.settings.prefillInt8 ? .on : .off
        int8Check = int8
        let checks = NSStackView(views: [ane, int8])
        checks.spacing = 16
        cfg.addArrangedSubview(checks)

        startButton = NSButton(title: "▶  Iniciar", target: self, action: #selector(startTapped))
        startButton?.bezelStyle = .rounded
        stopButton = NSButton(title: "⏹  Detener", target: self, action: #selector(stopTapped))
        stopButton?.bezelStyle = .rounded
        releaseButton = NSButton(title: "Liberar memoria", target: self, action: #selector(releaseTapped))
        releaseButton?.bezelStyle = .rounded
        let buttons = NSStackView(views: [startButton!, stopButton!, releaseButton!])
        buttons.spacing = 10
        cfg.addArrangedSubview(buttons)

        cfgBox.contentView = cfg
        cfgBox.translatesAutoresizingMaskIntoConstraints = false
        cfgBox.widthAnchor.constraint(equalToConstant: 608).isActive = true
        root.addArrangedSubview(cfgBox)

        // Live panel.
        let liveBox = NSBox()
        liveBox.title = "En vivo"
        liveBox.titlePosition = .atTop
        let liveGrid = NSStackView()
        liveGrid.orientation = .vertical
        liveGrid.spacing = 6
        liveGrid.translatesAutoresizingMaskIntoConstraints = true
        let keys = [
            "decode": "Decode",
            "prefill": "Prefill",
            "draft": "Draft",
            "requests": "Peticiones",
            "scheduler": "Scheduler",
            "kv": "KV cache",
            "cache": "Caché",
            "memory": "Memoria",
            "thermal": "Térmica",
            "weights": "Pesos",
            "plan": "Plan",
        ]
        for (k, title) in keys {
            let (view, value) = labelRow(title)
            live[k] = value
            liveGrid.addArrangedSubview(view)
        }
        liveBox.contentView = liveGrid
        liveBox.translatesAutoresizingMaskIntoConstraints = false
        liveBox.widthAnchor.constraint(equalToConstant: 608).isActive = true
        root.addArrangedSubview(liveBox)

        // In-flight requests (capped; the engine rarely runs many at once).
        let flightBox = NSBox()
        flightBox.title = "En vuelo"
        flightBox.titlePosition = .atTop
        inFlightStack = NSStackView()
        inFlightStack?.orientation = .vertical
        inFlightStack?.spacing = 4
        inFlightStack?.translatesAutoresizingMaskIntoConstraints = true
        let placeholder = NSTextField(labelWithString: "sin peticiones en curso")
        placeholder.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        placeholder.textColor = .tertiaryLabelColor
        inFlightStack?.addArrangedSubview(placeholder)
        flightBox.contentView = inFlightStack
        flightBox.translatesAutoresizingMaskIntoConstraints = false
        flightBox.widthAnchor.constraint(equalToConstant: 608).isActive = true
        root.addArrangedSubview(flightBox)

        // Footer.
        let logs = NSButton(title: "Abrir logs", target: self, action: #selector(openLogs))
        logs.bezelStyle = .rounded
        logs.controlSize = .small
        let footer = NSStackView(views: [logs])
        root.addArrangedSubview(footer)

        let container = NSView()
        container.addSubview(root)
        root.translatesAutoresizingMaskIntoConstraints = false
        root.topAnchor.constraint(equalTo: container.topAnchor).isActive = true
        root.bottomAnchor.constraint(equalTo: container.bottomAnchor).isActive = true
        root.leadingAnchor.constraint(equalTo: container.leadingAnchor).isActive = true
        root.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor).isActive = true
        w.contentView = container
        w.center()
        w.makeKeyAndOrderFront(nil)
        window = w
    }

    @discardableResult
    private func controlRow(_ label: String, _ make: () -> NSView) -> NSView {
        let l = NSTextField(labelWithString: label)
        l.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        l.textColor = .secondaryLabelColor
        l.frame.size.width = 110
        let v = make()
        let row = NSStackView(views: [l, v])
        row.spacing = 8
        return row
    }

    // MARK: Actions

    @objc private func copyAPI() {
        let s = config.settings
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("http://127.0.0.1:\(s.port)/v1", forType: .string)
    }

    @objc private func openLogs() {
        NSWorkspace.shared.open(Config.logFileURL())
    }

    @objc private func modelChanged() {
        guard let p = modelPopup, let m = modelsList[safe: p.indexOfSelectedItem] else { return }
        config.settings.model = m.id
        config.save()
        confirmRestartIfNeeded()
    }

    @objc private func portChanged() {
        config.settings.port = Int(portField?.stringValue ?? "") ?? config.settings.port
        config.save()
        engine.port = config.settings.port
        apiLabel?.stringValue = "API  http://127.0.0.1:\(config.settings.port)/v1"
        confirmRestartIfNeeded()
    }

    @objc private func settingsChanged() {
        readControls()
        config.save()
        confirmRestartIfNeeded()
    }

    private var modelsList: [(id: String, sizeGB: Double)] {
        engine.models(root: config.settings.splashRoot)
    }

    private func reloadModels() {
        let list = modelsList
        let p = modelPopup!
        p.removeAllItems()
        for m in list {
            p.addItem(withTitle: String(format: "%@  (%.1f GB)", m.id, m.sizeGB))
        }
        p.addItems(withTitles: ["(elegí un modelo)"])
    }

    private func readControls() {
        var s = config.settings
        if let m = modelPopup, let item = modelsList[safe: m.indexOfSelectedItem], item.id != "(elegí un modelo)" {
            s.model = item.id
        }
        s.port = Int(portField?.stringValue ?? "") ?? s.port
        s.context = ["16K", "32K", "64K", "128K", "auto"][ctxSegment?.selectedSegment ?? 4]
        s.kvFormat = ["int8", "bf16"][kvSegment?.selectedSegment ?? 0]
        s.idleRelease = ["off", "10m", "30m", "2h"][idleSegment?.selectedSegment ?? 0]
        s.maxMemory = ["auto", "24G", "28G", "32G"][memSegment?.selectedSegment ?? 0]
        s.cacheDisk = ["auto", "16G", "off"][cacheSegment?.selectedSegment ?? 0]
        s.useANE = aneCheck?.state == .on
        s.prefillInt8 = int8Check?.state == .on
        config.settings = s
        engine.port = s.port
    }

    private func syncControls() {
        let s = config.settings
        let list = modelsList
        if let p = modelPopup {
            for (i, m) in list.enumerated() where m.id == s.model {
                p.selectItem(at: i)
                break
            }
        }
        portField?.stringValue = "\(s.port)"
        func index(_ values: [String], _ value: String) -> Int {
            values.firstIndex(of: value) ?? 0
        }
        ctxSegment?.selectedSegment = index(["16K", "32K", "64K", "128K", "auto"], s.context)
        kvSegment?.selectedSegment = index(["int8", "bf16"], s.kvFormat)
        idleSegment?.selectedSegment = index(["off", "10m", "30m", "2h"], s.idleRelease)
        memSegment?.selectedSegment = index(["auto", "24G", "28G", "32G"], s.maxMemory)
        cacheSegment?.selectedSegment = index(["auto", "16G", "off"], s.cacheDisk)
        aneCheck?.state = s.useANE ? .on : .off
        int8Check?.state = s.prefillInt8 ? .on : .off
    }

    private func confirmRestartIfNeeded() {
        guard engine.state != .stopped else { tick(); return }
        let alert = NSAlert()
        alert.messageText = "Reiniciar el engine?"
        alert.informativeText = "Los cambios se aplican al reiniciar. Las peticiones en curso se interrumpen."
        alert.addButton(withTitle: "Reiniciar")
        alert.addButton(withTitle: "Cancelar")
        guard alert.runModal() == .alertFirstButtonReturn else {
            syncControls()
            return
        }
        engine.stop()
        do {
            try engine.start(root: config.settings.splashRoot, model: config.settings.model, s: config.settings)
        } catch {
            let err = NSAlert()
            err.messageText = "No pudo reiniciar"
            err.informativeText = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            err.runModal()
        }
        tick()
    }

    @objc private func startTapped() {
        do {
            try engine.start(root: config.settings.splashRoot, model: config.settings.model, s: config.settings)
        } catch {
            let alert = NSAlert()
            alert.messageText = "No pudo iniciar"
            alert.informativeText = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            alert.runModal()
        }
        tick()
    }

    @objc private func stopTapped() {
        engine.stop()
        tick()
    }

    @objc private func releaseTapped() {
        engine.releaseMemory(config.settings.port)
    }

    // MARK: Live tick

    private func tick() {
        let s = config.settings
        let state = engine.state
        let dot: NSColor
        let stateWord: String
        switch state {
        case .stopped: (dot, stateWord) = (.systemGray, "detenido")
        case .starting: (dot, stateWord) = (.systemYellow, "cargando…")
        case .running:
            let pressure = engine.lastStatus?["memory_pressure"] as? String
            dot = pressure == "critical" ? .systemRed : pressure == "warning" ? .systemOrange : .systemGreen
            stateWord = "activo"
        }
        statusItem?.button?.contentTintColor = dot
        headerValue?.stringValue = "  \(stateWord)  ·  \(s.model)  ·  :\(s.port)"
        headerValue?.textColor = dot
        startButton?.isEnabled = state == .stopped
        stopButton?.isEnabled = state != .stopped
        releaseButton?.isEnabled = state == .running

        let st = engine.lastStatus
        func set(_ key: String, _ value: String, _ color: NSColor = .labelColor) {
            guard let field = live[key] else { return }
            field.stringValue = st == nil && state != .running ? "—" : value
            field.textColor = color
        }

        if let st {
            let metrics = st["metrics"] as? [String: Any]
            if let m = metrics, let tps = m["decode_tokens_per_second"] as? Double, tps > 0 {
                var text = String(format: "%.1f tok/s", tps)
                if let itl = m["itl_ms"] as? [String: Any], let p50 = itl["p50"] as? Double {
                    text += "   ·   ITL " + Fmt.ms(p50)
                }
                set("decode", text, .systemGreen)
            } else {
                set("decode", "en reposo")
            }
            if let m = metrics, let tps = m["prefill_tokens_per_second"] as? Double, tps > 0 {
                var text = String(format: "%.0f tok/s", tps)
                if let ttft = m["ttft_ms"] as? [String: Any], let p50 = ttft["p50"] as? Double {
                    text += "   ·   TTFT " + Fmt.ms(p50)
                }
                set("prefill", text)
            } else {
                set("prefill", "—")
            }
            if let m = metrics,
               let drafted = m["drafted_tokens"] as? Double, drafted > 0,
               let accepted = m["accepted_draft_tokens"] as? Double {
                set("draft", String(format: "%.0f%% accept   (%@/%@)", accepted / drafted * 100,
                                     Fmt.tokD(accepted), Fmt.tokD(drafted)))
            } else {
                set("draft", "—")
            }
            if let r = st["requests"] as? [String: Any] {
                let sub = r["submitted"] as? Int ?? 0
                let done = r["completed"] as? Int ?? 0
                let cancel = r["cancelled"] as? Int ?? 0
                let failed = r["failed"] as? Int ?? 0
                set("requests", "\(sub) enviadas   ·   \(done) hechas"
                    + (cancel > 0 ? "   ·   \(cancel) cancel" : "")
                    + (failed > 0 ? "   ·   \(failed) fallidas" : ""))
            }
            if let sch = st["scheduler"] as? [String: Any] {
                set("scheduler", "\(sch["decoding"] as? Int ?? 0) decodificando   ·   "
                    + "\(sch["prefilling"] as? Int ?? 0) en prefill   ·   "
                    + "\(sch["queued"] as? Int ?? 0) en cola")
            }
            if let kv = st["kv"] as? [String: Any] {
                var parts: [String] = []
                if let a = kv["pages_active"] as? Int, let c = kv["pages_cache"] as? Int,
                   let total = kv["pages_allocated"] as? Int {
                    parts.append("\(a + c)/\(total) pág")
                }
                if let b = kv["allocated_bytes"] as? Double { parts.append(Fmt.bytes(b)) }
                if let block = kv["block_tokens"] as? Int, let a = kv["pages_active"] as? Int {
                    let maxTok = st["maximum_context_tokens"] as? Int
                    var ctx = "\(Fmt.tok(a * block)) tok"
                    if let maxTok { ctx += " / " + Fmt.tok(maxTok) }
                    parts.append(ctx)
                }
                set("kv", parts.joined(separator: "   ·   "))
            }
            if let cache = st["cache"] as? [String: Any] {
                var parts: [String] = []
                if let rate = cache["hit_rate"] as? Double {
                    parts.append(String(format: "%.0f%% hit", rate * 100))
                }
                if let reused = cache["reused_tokens"] as? Double {
                    parts.append(Fmt.tokD(reused) + " tok reusados")
                }
                if let disk = st["disk"] as? [String: Any],
                   let used = disk["used_bytes"] as? Double, let cap = disk["capacity_bytes"] as? Double {
                    parts.append("disco " + Fmt.bytes(used) + "/" + Fmt.bytes(cap))
                }
                set("cache", parts.joined(separator: "   ·   "))
            }
            if let mem = st["memory_actual"] as? [String: Any], let a = mem["allocated_bytes"] as? Double {
                var text = Fmt.bytes(a)
                if let gov = st["memory_governor"] as? [String: Any], let limit = gov["limit_bytes"] as? Double, limit > 0 {
                    text += " / " + Fmt.bytes(limit)
                }
                let pressure = st["memory_pressure"] as? String
                if let pressure { text += "   ·   " + pressure }
                let color: NSColor = pressure == "critical" ? .systemRed
                    : pressure == "warning" ? .systemOrange : .labelColor
                set("memory", text, color)
            }
            if let thermal = st["thermal_state"] as? String {
                let color: NSColor = thermal == "hot" ? .systemOrange : .labelColor
                set("thermal", thermal, color)
            } else {
                set("thermal", "—")
            }
            if let weights = st["weights"] as? [String: Any] {
                if weights["released"] as? Bool == true {
                    set("weights", "liberados (vuelven bajo demanda)", .systemOrange)
                } else if weights["restoring"] as? Bool == true {
                    set("weights", "restaurando…", .systemOrange)
                } else {
                    var text = "cargados"
                    if let sec = weights["idle_release_seconds"] as? Double, sec > 0 {
                        text += "   ·   libera tras \(Int(sec / 60)) min inactivo"
                    }
                    set("weights", text)
                }
            }
            if let plan = st["memory_plan"] as? [String: Any] {
                let device = plan["device"] as? [String: Any]
                var text = "\(device?["device_name"] as? String ?? "?")   ·   "
                    + "\(Int((device?["physical_memory_bytes"] as? Double ?? 0) / 1_000_000_000)) GB"
                if let model = plan["model"] as? [String: Any], let ctx = model["maximum_context_tokens"] as? Int {
                    text += "   ·   ctx máx " + Fmt.tok(ctx)
                }
                set("plan", text)
            }
            updateInFlight(st)
        } else {
            for field in live.values {
                field.stringValue = "—"
                field.textColor = .tertiaryLabelColor
            }
            inFlightKey = "\u{0}"   // force a rebuild on the next tick
            if let stack = inFlightStack {
                stack.arrangedSubviews.forEach { $0.removeFromSuperview() }
                let ph = NSTextField(labelWithString: state == .running ? "sin datos" : "engine detenido")
                ph.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
                ph.textColor = .tertiaryLabelColor
                stack.addArrangedSubview(ph)
            }
        }
    }

    private let maxInFlightRows = 6

    private func updateInFlight(_ st: [String: Any]) {
        guard let stack = inFlightStack else { return }
        let active = st["active_requests"] as? [[String: Any]] ?? []
        let key = active.map { "\(($0["id"] as? Int) ?? 0)" }.joined(separator: ",")
        let visible = Array(active.prefix(maxInFlightRows))
        let expected = active.isEmpty ? 1 : visible.count + (active.count > maxInFlightRows ? 1 : 0)

        guard key == inFlightKey, stack.arrangedSubviews.count == expected else {
            inFlightKey = key
            stack.arrangedSubviews.forEach { $0.removeFromSuperview() }
            if visible.isEmpty {
                let ph = NSTextField(labelWithString: "sin peticiones en curso")
                ph.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
                ph.textColor = .tertiaryLabelColor
                stack.addArrangedSubview(ph)
                return
            }
            for row in visible {
                let l = NSTextField(labelWithString: inFlightLine(row))
                l.font = .monospacedDigitSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular)
                stack.addArrangedSubview(l)
            }
            if active.count > maxInFlightRows {
                let more = NSTextField(labelWithString: "+\(active.count - maxInFlightRows) más")
                more.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
                more.textColor = .secondaryLabelColor
                stack.addArrangedSubview(more)
            }
            return
        }

        // Same shape: refresh values in place.
        for (i, row) in visible.enumerated() {
            guard let label = stack.arrangedSubviews[i] as? NSTextField,
                  label.stringValue.hasPrefix("#") else { continue }
            label.stringValue = inFlightLine(row)
        }
    }

    private func inFlightLine(_ r: [String: Any]) -> String {
        let id = r["id"] as? Int ?? 0
        let phase = r["phase"] as? String ?? "?"
        let prompt = r["prompt_tokens"] as? Int ?? 0
        let processed = r["prompt_processed"] as? Int ?? 0
        let generated = r["generated_tokens"] as? Int ?? 0
        let maxNew = r["max_new_tokens"] as? Int ?? 0
        let age = Int((r["age_ms"] as? Double ?? 0) / 1000)
        return "#\(id)   \(phase)   ·   prompt \(processed)/\(prompt)"
            + (maxNew > 0 ? "   ·   out \(generated)/\(maxNew)" : "   ·   out \(generated)")
            + "   ·   \(age)s"
    }
}

extension Sequence {
    subscript(safe index: Int) -> Element? {
        var i = index
        var iterator = makeIterator()
        while i >= 0, iterator.next() != nil { i -= 1 }
        guard i == 0 else { return nil }
        return iterator.next()
    }
}

// MARK: - Entry point

let arguments = Array(CommandLine.arguments.dropFirst())
if !arguments.isEmpty {
    runCLI(arguments)
    exit(0)
}

let app = NSApplication.shared
let delegate = AppController()
app.delegate = delegate
app.run()
