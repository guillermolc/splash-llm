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

// MARK: - Palette (MTPLX "hippo")

enum Palette {
    static let background = NSColor(srgbRed: 0.063, green: 0.082, blue: 0.106, alpha: 1)
    static let card = NSColor(white: 1, alpha: 0.05)
    static let cardBorder = NSColor(white: 1, alpha: 0.08)
    static let label = NSColor(srgbRed: 0.55, green: 0.58, blue: 0.62, alpha: 1)
    static let value = NSColor(white: 0.96, alpha: 1)
    static let sub = NSColor(white: 0.55, alpha: 1)
    static let green = NSColor(srgbRed: 0.0, green: 0.84, blue: 0.56, alpha: 1)   // #00d68f
    static let orange = NSColor(srgbRed: 0.98, green: 0.61, blue: 0.0, alpha: 1)  // #f99c00
    static let red = NSColor(srgbRed: 0.94, green: 0.35, blue: 0.42, alpha: 1)    // #f0586a
    static let blue = NSColor(srgbRed: 0.31, green: 0.71, blue: 0.95, alpha: 1)   // #4fb6f3
    static let dim = NSColor(white: 0.35, alpha: 1)
}

// MARK: - CLI one-shots (smoke-test the HTTP paths without a GUI)

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
    case "--stop":
        let port = args.count > 1 ? (Int(args[1]) ?? cfg.settings.port) : cfg.settings.port
        let e = Engine()
        e.port = port
        e.adoptIfRunning()
        guard let inst = e.lastStatus?["instance"] as? [String: Any],
              let v = inst["pid"] as? Int else {
            print("no engine corriendo en :\(port)")
            exit(1)
        }
        e.killTreeSync(Int32(v))
        print("engine :\(port) detenido")
    case "--release":
        let port = args.count > 1 ? (Int(args[1]) ?? cfg.settings.port) : cfg.settings.port
        Engine().releaseMemory(port)
        print("pedido de liberación enviado a :\(port)")
    case "--logs":
        NSWorkspace.shared.open(Config.logFileURL())
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

// MARK: - UI building blocks

/// A rounded, semi-transparent metric card: small caps label, big mono value, small sub.
func metricCard(_ title: String) -> (view: NSView, main: NSTextField, sub: NSTextField) {
    let l = NSTextField(labelWithString: title.uppercased())
    l.font = .systemFont(ofSize: 9, weight: .semibold)
    l.textColor = Palette.label

    let main = NSTextField(labelWithString: "—")
    main.font = .monospacedDigitSystemFont(ofSize: 17, weight: .semibold)
    main.textColor = Palette.value

    let sub = NSTextField(labelWithString: "")
    sub.font = .monospacedDigitSystemFont(ofSize: 10, weight: .regular)
    sub.textColor = Palette.sub

    let stack = NSStackView(views: [l, main, sub])
    stack.orientation = .vertical
    stack.alignment = .leading
    stack.spacing = 3
    stack.translatesAutoresizingMaskIntoConstraints = false

    let card = NSView()
    card.wantsLayer = true
    card.layer?.backgroundColor = Palette.card.cgColor
    card.layer?.borderColor = Palette.cardBorder.cgColor
    card.layer?.borderWidth = 1
    card.layer?.cornerRadius = 10
    card.addSubview(stack)
    stack.topAnchor.constraint(equalTo: card.topAnchor, constant: 10).isActive = true
    stack.bottomAnchor.constraint(equalTo: card.bottomAnchor, constant: -10).isActive = true
    stack.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 12).isActive = true
    stack.trailingAnchor.constraint(lessThanOrEqualTo: card.trailingAnchor, constant: -12).isActive = true
    return (card, main, sub)
}

func sectionHeader(_ text: String) -> NSTextField {
    let l = NSTextField(labelWithString: text.uppercased())
    l.font = .systemFont(ofSize: 10, weight: .semibold)
    l.textColor = Palette.label
    return l
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

    // Live cards.
    var cards: [String: (main: NSTextField, sub: NSTextField)] = [:]
    var ctxBarFill: NSView?
    var ctxBarWidth: NSLayoutConstraint?
    var ctxBarLabel: NSTextField?
    var headerValue: NSTextField?
    var apiLabel: NSTextField?
    var inFlightStack: NSStackView?
    var inFlightKey: String = ""

    var startButton: NSButton?
    var stopButton: NSButton?
    var releaseButton: NSButton?

    private let maxInFlightRows = 6

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.appearance = NSAppearance(named: .darkAqua)
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
            contentRect: NSRect(x: 0, y: 0, width: 680, height: 760),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        w.title = "Splash"
        w.backgroundColor = Palette.background
        w.isReleasedWhenClosed = false
        w.delegate = self
        w.minSize = NSSize(width: 560, height: 520)

        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 14
        root.edgeInsets = NSEdgeInsets(top: 18, left: 18, bottom: 18, right: 18)

        // ── Header ─────────────────────────────────────────────
        let dot = NSTextField(labelWithString: "●")
        dot.font = .systemFont(ofSize: 15)
        let title = NSTextField(labelWithString: "SPLASH")
        title.font = .boldSystemFont(ofSize: 14)
        title.textColor = Palette.value
        let hv = NSTextField(labelWithString: "")
        hv.font = .systemFont(ofSize: 12)
        hv.textColor = Palette.sub
        headerValue = hv
        let header = NSStackView(views: [dot, title, hv])
        header.spacing = 8
        let api = NSTextField(labelWithString: "API  http://127.0.0.1:\(config.settings.port)/v1")
        api.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        api.textColor = Palette.sub
        apiLabel = api
        let copyBtn = NSButton(title: "copiar", target: self, action: #selector(copyAPI))
        copyBtn.bezelStyle = .rounded
        copyBtn.controlSize = .small
        let apiRow = NSStackView(views: [api, copyBtn])
        apiRow.spacing = 8
        let headerCol = NSStackView(views: [header, apiRow])
        headerCol.orientation = .vertical
        headerCol.spacing = 5
        root.addArrangedSubview(headerCol)

        // ── Live metric cards (3 × 4) ──────────────────────────
        let liveGrid = NSStackView()
        liveGrid.orientation = .vertical
        liveGrid.spacing = 10
        liveGrid.distribution = .fill
        let keys = [
            ["decode", "prefill", "draft"],
            ["requests", "scheduler", "kv"],
            ["cache", "memory", "context"],
            ["weights", "thermal", "plan"],
        ]
        for row in keys {
            let rowStack = NSStackView()
            rowStack.orientation = .horizontal
            rowStack.distribution = .fillEqually
            rowStack.spacing = 10
            for key in row {
                if key == "context" {
                    let c = contextCard()
                    ctxBarFill = c.fill
                    ctxBarWidth = c.width
                    ctxBarLabel = c.label
                    rowStack.addArrangedSubview(c.view)
                    continue
                }
                let card = metricCard(key)
                cards[key] = (card.main, card.sub)
                rowStack.addArrangedSubview(card.view)
            }
            liveGrid.addArrangedSubview(rowStack)
        }
        root.addArrangedSubview(liveGrid)

        // ── Configuration ──────────────────────────────────────
        root.addArrangedSubview(sectionHeader("Modelo y ajustes"))
        let cfg = NSStackView()
        cfg.orientation = .vertical
        cfg.spacing = 10

        let row1 = NSStackView(views: [
            controlField("Modelo") {
                let p = NSPopUpButton()
                p.target = self
                p.action = #selector(modelChanged)
                self.modelPopup = p
                return p
            },
            controlField("Puerto") {
                let f = NSTextField(string: "\(config.settings.port)")
                f.widthAnchor.constraint(equalToConstant: 64).isActive = true
                f.target = self
                f.action = #selector(portChanged)
                self.portField = f
                return f
            },
        ])
        row1.distribution = .fill
        row1.spacing = 12
        cfg.addArrangedSubview(row1)

        let row2 = NSStackView(views: [
            controlField("Contexto") {
                let s = NSSegmentedControl(labels: ["16K", "32K", "64K", "128K", "auto"],
                                           trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
                self.ctxSegment = s
                return s
            },
            controlField("KV") {
                let s = NSSegmentedControl(labels: ["int8", "bf16"],
                                           trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
                self.kvSegment = s
                return s
            },
        ])
        row2.distribution = .fill
        row2.spacing = 12
        cfg.addArrangedSubview(row2)

        let row3 = NSStackView(views: [
            controlField("Liberar inactivo") {
                let s = NSSegmentedControl(labels: ["off", "10m", "30m", "2h"],
                                           trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
                self.idleSegment = s
                return s
            },
            controlField("Techo memoria") {
                let s = NSSegmentedControl(labels: ["auto", "24G", "28G", "32G"],
                                           trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
                self.memSegment = s
                return s
            },
            controlField("Caché SSD") {
                let s = NSSegmentedControl(labels: ["auto", "16G", "off"],
                                           trackingMode: .selectOne, target: self, action: #selector(settingsChanged))
                self.cacheSegment = s
                return s
            },
        ])
        row3.distribution = .fill
        row3.spacing = 12
        cfg.addArrangedSubview(row3)

        let ane = NSButton(checkboxWithTitle: "Neural Engine (ANE)", target: self, action: #selector(settingsChanged))
        ane.state = config.settings.useANE ? .on : .off
        aneCheck = ane
        let int8 = NSButton(checkboxWithTitle: "Prefill int8", target: self, action: #selector(settingsChanged))
        int8.state = config.settings.prefillInt8 ? .on : .off
        int8Check = int8
        let checks = NSStackView(views: [ane, int8])
        checks.spacing = 20
        cfg.addArrangedSubview(checks)

        startButton = NSButton(title: "▶  Iniciar", target: self, action: #selector(startTapped))
        stopButton = NSButton(title: "⏹  Detener", target: self, action: #selector(stopTapped))
        releaseButton = NSButton(title: "Liberar memoria", target: self, action: #selector(releaseTapped))
        for b in [startButton, stopButton, releaseButton] { b?.bezelStyle = .rounded }
        let buttons = NSStackView(views: [startButton!, stopButton!, releaseButton!])
        buttons.spacing = 10
        cfg.addArrangedSubview(buttons)
        root.addArrangedSubview(cfg)

        // ── In-flight ──────────────────────────────────────────
        root.addArrangedSubview(sectionHeader("En vuelo"))
        let flight = NSView()
        flight.wantsLayer = true
        flight.layer?.backgroundColor = Palette.card.cgColor
        flight.layer?.borderColor = Palette.cardBorder.cgColor
        flight.layer?.borderWidth = 1
        flight.layer?.cornerRadius = 10
        inFlightStack = NSStackView()
        inFlightStack?.orientation = .vertical
        inFlightStack?.spacing = 6
        inFlightStack?.translatesAutoresizingMaskIntoConstraints = true
        addFlightPlaceholder()
        flight.addSubview(inFlightStack!)
        inFlightStack!.topAnchor.constraint(equalTo: flight.topAnchor, constant: 10).isActive = true
        inFlightStack!.leadingAnchor.constraint(equalTo: flight.leadingAnchor, constant: 12).isActive = true
        inFlightStack!.trailingAnchor.constraint(lessThanOrEqualTo: flight.trailingAnchor, constant: -12).isActive = true
        inFlightStack!.bottomAnchor.constraint(equalTo: flight.bottomAnchor, constant: -10).isActive = true
        flight.widthAnchor.constraint(equalToConstant: 644).isActive = true
        root.addArrangedSubview(flight)

        // ── Footer ─────────────────────────────────────────────
        let logs = NSButton(title: "Abrir logs", target: self, action: #selector(openLogs))
        logs.bezelStyle = .rounded
        logs.controlSize = .small
        root.addArrangedSubview(logs)

        // Fit the window to its content.
        let content = NSView()
        content.addSubview(root)
        root.translatesAutoresizingMaskIntoConstraints = false
        root.topAnchor.constraint(equalTo: content.topAnchor).isActive = true
        root.bottomAnchor.constraint(equalTo: content.bottomAnchor).isActive = true
        root.leadingAnchor.constraint(equalTo: content.leadingAnchor).isActive = true
        root.trailingAnchor.constraint(lessThanOrEqualTo: content.trailingAnchor).isActive = true
        w.contentView = content
        w.center()
        w.makeKeyAndOrderFront(nil)
        window = w
    }

    @discardableResult
    private func controlField(_ label: String, _ make: () -> NSView) -> NSView {
        let l = NSTextField(labelWithString: label.uppercased())
        l.font = .systemFont(ofSize: 9, weight: .semibold)
        l.textColor = Palette.label
        l.frame.size.width = 100
        let v = make()
        let row = NSStackView(views: [l, v])
        row.spacing = 8
        return row
    }

    private func contextCard() -> (view: NSView, fill: NSView, width: NSLayoutConstraint, label: NSTextField) {
        let (card, main, sub) = metricCard("Contexto")
        main.stringValue = "—"
        sub.stringValue = ""
        let bar = NSView()
        bar.wantsLayer = true
        bar.layer?.backgroundColor = NSColor(white: 1, alpha: 0.1).cgColor
        bar.layer?.cornerRadius = 4
        let fill = NSView()
        fill.wantsLayer = true
        fill.layer?.backgroundColor = Palette.green.cgColor
        fill.layer?.cornerRadius = 4
        fill.translatesAutoresizingMaskIntoConstraints = false
        bar.addSubview(fill)
        fill.topAnchor.constraint(equalTo: bar.topAnchor).isActive = true
        fill.bottomAnchor.constraint(equalTo: bar.bottomAnchor).isActive = true
        fill.leadingAnchor.constraint(equalTo: bar.leadingAnchor).isActive = true
        let width = fill.widthAnchor.constraint(equalToConstant: 0)
        width.isActive = true
        bar.heightAnchor.constraint(equalToConstant: 8).isActive = true
        let stack = card.subviews.first as! NSStackView
        stack.addArrangedSubview(bar)
        bar.widthAnchor.constraint(equalTo: card.widthAnchor, constant: -24).isActive = true
        cards["context"] = (main, sub)
        return (card, fill, width, main)
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
    }

    private func readControls() {
        var s = config.settings
        if let m = modelPopup, let item = modelsList[safe: m.indexOfSelectedItem] {
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

    private func setCard(_ key: String, main: String, sub: String = "",
                         color: NSColor? = nil, subColor: NSColor? = nil) {
        guard let c = cards[key] else { return }
        c.main.stringValue = main
        c.main.textColor = color ?? Palette.value
        c.sub.stringValue = sub
        c.sub.textColor = subColor ?? Palette.sub
    }

    private func tick() {
        let s = config.settings
        let state = engine.state
        let dot: NSColor
        let stateWord: String
        switch state {
        case .stopped: (dot, stateWord) = (Palette.dim, "detenido")
        case .starting: (dot, stateWord) = (Palette.orange, "cargando…")
        case .running:
            let pressure = engine.lastStatus?["memory_pressure"] as? String
            dot = pressure == "critical" ? Palette.red : pressure == "warning" ? Palette.orange : Palette.green
            stateWord = "activo"
        }
        statusItem?.button?.contentTintColor = dot
        (headerValue)?.stringValue = "   \(stateWord)  ·  \(s.model)  ·  :\(s.port)"
        headerValue?.textColor = Palette.sub

        startButton?.isEnabled = state == .stopped
        stopButton?.isEnabled = state != .stopped
        releaseButton?.isEnabled = state == .running
        // Start button accent.
        if let start = startButton, state == .stopped {
            start.contentTintColor = Palette.green
        } else {
            startButton?.contentTintColor = nil
        }

        let st = engine.lastStatus
        if let st {
            if let m = st["metrics"] as? [String: Any] {
                // decode
                if let tps = m["decode_tokens_per_second"] as? Double, tps > 0 {
                    var sub = ""
                    if let itl = m["itl_ms"] as? [String: Any], let p50 = itl["p50"] as? Double {
                        sub = "ITL " + Fmt.ms(p50)
                    }
                    setCard("decode", main: String(format: "%.1f tok/s", tps), sub: sub, color: Palette.green)
                } else {
                    setCard("decode", main: "reposo", sub: "")
                }
                // prefill
                if let tps = m["prefill_tokens_per_second"] as? Double, tps > 0 {
                    var sub = ""
                    if let ttft = m["ttft_ms"] as? [String: Any], let p50 = ttft["p50"] as? Double {
                        sub = "TTFT " + Fmt.ms(p50)
                    }
                    setCard("prefill", main: String(format: "%.0f tok/s", tps), sub: sub, color: Palette.blue)
                } else {
                    setCard("prefill", main: "—")
                }
                // draft
                if let drafted = m["drafted_tokens"] as? Double, drafted > 0,
                   let accepted = m["accepted_draft_tokens"] as? Double {
                    setCard("draft", main: String(format: "%.0f%%", accepted / drafted * 100),
                            sub: "accept  " + Fmt.tokD(accepted) + "/" + Fmt.tokD(drafted),
                            color: Palette.green)
                } else {
                    setCard("draft", main: "—", sub: "sin draft")
                }
            }
            if let r = st["requests"] as? [String: Any] {
                let sub = r["submitted"] as? Int ?? 0
                let done = r["completed"] as? Int ?? 0
                let cancel = r["cancelled"] as? Int ?? 0
                let failed = r["failed"] as? Int ?? 0
                setCard("requests", main: "\(sub)",
                        sub: "\(done) hechas" + (cancel > 0 ? " · \(cancel) cancel" : "")
                            + (failed > 0 ? " · \(failed) fallidas" : ""))
            }
            if let sch = st["scheduler"] as? [String: Any] {
                let d = sch["decoding"] as? Int ?? 0
                let p = sch["prefilling"] as? Int ?? 0
                let q = sch["queued"] as? Int ?? 0
                setCard("scheduler", main: "\(d) · \(p) · \(q)", sub: "decode · prefill · cola")
            }
            if let kv = st["kv"] as? [String: Any] {
                var main = "—"
                var sub = ""
                if let b = kv["allocated_bytes"] as? Double { main = Fmt.bytes(b) }
                var parts: [String] = []
                if let a = kv["pages_active"] as? Int, let c = kv["pages_cache"] as? Int,
                   let total = kv["pages_allocated"] as? Int, total > 0 {
                    parts.append("\(a + c)/\(total) pág")
                    if let block = kv["block_tokens"] as? Int {
                        let used = (a + c) * block
                        let maxTok = st["maximum_context_tokens"] as? Int ?? 0
                        if maxTok > 0 {
                            let pct = Double(used) / Double(maxTok) * 100
                            if let bar = ctxBarFill {
                                ctxBarWidth?.constant = bar.bounds.width * pct / 100
                            }
                            ctxBarLabel?.stringValue = String(format: "%.0f%%", pct)
                            ctxBarLabel?.textColor = pct > 85 ? Palette.orange : Palette.value
                            parts.append("ctx " + Fmt.tok(used) + "/" + Fmt.tok(maxTok))
                        }
                    }
                }
                if !parts.isEmpty { sub = parts.joined(separator: " · ") }
                setCard("kv", main: main, sub: sub)
            }
            if let cache = st["cache"] as? [String: Any] {
                var parts: [String] = []
                var main = "—"
                if let rate = cache["hit_rate"] as? Double {
                    main = String(format: "%.0f%%", rate * 100)
                }
                if let reused = cache["reused_tokens"] as? Double { parts.append(Fmt.tokD(reused) + " reusados") }
                if let disk = st["disk"] as? [String: Any],
                   let used = disk["used_bytes"] as? Double, let cap = disk["capacity_bytes"] as? Double, cap > 0 {
                    parts.append("SSD " + Fmt.bytes(used) + "/" + Fmt.bytes(cap))
                }
                setCard("cache", main: main, sub: parts.joined(separator: " · "), color: main == "—" ? nil : Palette.blue)
            }
            if let mem = st["memory_actual"] as? [String: Any], let a = mem["allocated_bytes"] as? Double {
                var main = Fmt.bytes(a)
                var sub = ""
                if let gov = st["memory_governor"] as? [String: Any], let limit = gov["limit_bytes"] as? Double, limit > 0 {
                    main += " / " + Fmt.bytes(limit)
                }
                let pressure = st["memory_pressure"] as? String
                if let pressure { sub = "presión " + pressure }
                let c: NSColor = pressure == "critical" ? Palette.red
                    : pressure == "warning" ? Palette.orange : Palette.value
                setCard("memory", main: main, sub: sub, color: c)
            }
            if let thermal = st["thermal_state"] as? String {
                let c: NSColor = thermal == "hot" ? Palette.orange : Palette.value
                setCard("thermal", main: thermal, color: c)
            } else {
                setCard("thermal", main: "—", sub: "sin datos")
            }
            if let weights = st["weights"] as? [String: Any] {
                if weights["released"] as? Bool == true {
                    setCard("weights", main: "liberados", sub: "vuelven bajo demanda", color: Palette.orange)
                } else if weights["restoring"] as? Bool == true {
                    setCard("weights", main: "restaurando…", color: Palette.orange)
                } else {
                    var sub = ""
                    if let sec = weights["idle_release_seconds"] as? Double, sec > 0 {
                        sub = "libera tras \(Int(sec / 60)) min"
                    }
                    setCard("weights", main: "cargados", sub: sub, color: Palette.green)
                }
            }
            if let plan = st["memory_plan"] as? [String: Any] {
                let device = plan["device"] as? [String: Any]
                var main = device?["device_name"] as? String ?? "—"
                if let gb = device?["physical_memory_bytes"] as? Double {
                    main += "  ·  \(Int(gb / 1_000_000_000)) GB"
                }
                setCard("plan", main: main, sub: "plan del engine")
            }
            updateInFlight(st)
        } else {
            for (key, c) in cards where key != "context" {
                c.main.stringValue = "—"
                c.main.textColor = Palette.dim
                c.sub.stringValue = ""
            }
            if ctxBarWidth != nil {
                ctxBarWidth?.constant = (ctxBarFill?.bounds.width ?? 0) * 0
            }
            ctxBarLabel?.stringValue = "—"
            inFlightKey = "\u{0}"
            if let stack = inFlightStack {
                stack.arrangedSubviews.forEach { $0.removeFromSuperview() }
                addFlightPlaceholder(state == .running ? "sin datos" : "engine detenido")
            }
        }
    }

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
                addFlightPlaceholder()
                return
            }
            for row in visible {
                let l = NSTextField(labelWithString: inFlightLine(row))
                l.font = .monospacedDigitSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular)
                l.textColor = Palette.value
                stack.addArrangedSubview(l)
            }
            if active.count > maxInFlightRows {
                let more = NSTextField(labelWithString: "+\(active.count - maxInFlightRows) más")
                more.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
                more.textColor = Palette.sub
                stack.addArrangedSubview(more)
            }
            return
        }
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

    private func addFlightPlaceholder(_ text: String = "sin peticiones en curso") {
        let ph = NSTextField(labelWithString: text)
        ph.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        ph.textColor = Palette.dim
        inFlightStack?.addArrangedSubview(ph)
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
