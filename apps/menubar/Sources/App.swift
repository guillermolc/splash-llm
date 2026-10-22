import SwiftUI
import AppKit

@main
struct SplashStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("Splash", id: "main") {
            ContentView()
        }
        .defaultSize(width: 660, height: 720)
        .windowResizability(.contentMinSize)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // CLI one-shots: run and exit (no window).
        let args = CommandLine.arguments.dropFirst()
        if !args.isEmpty && args.first?.hasPrefix("--") == true {
            CLI.run(Array(args))
            NSApp.terminate(nil)
            return
        }

        NSApp.setActivationPolicy(.regular)
        NSApp.appearance = NSAppearance(named: .darkAqua)

        // Menu bar presence.
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = item.button {
            button.image = NSImage(systemSymbolName: "brain", accessibilityDescription: "Splash")
            button.image?.isTemplate = true
        }
        let menu = NSMenu()
        menu.addItem(ActionItem("Mostrar ventana") {
            NSApp.activate(ignoringOtherApps: true)
            self.openMainWindow()
        })
        menu.addItem(ActionItem("Detener engine") { Engine.shared.stop() })
        menu.addItem(.separator())
        menu.addItem(ActionItem("Salir") { NSApp.terminate(nil) })
        item.menu = menu
        statusItem = item

        // Main menu (⌘Q).
        let main = NSMenu()
        let appItem = NSMenuItem()
        main.addItem(appItem)
        let salir = NSMenuItem(title: "Salir", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        salir.target = NSApp
        let appMenu = NSMenu()
        appMenu.addItem(salir)
        appItem.submenu = appMenu
        NSApp.mainMenu = main

        Engine.shared.adoptIfRunning()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func openMainWindow() {
        for window in NSApp.windows where window.title == "Splash" {
            window.makeKeyAndOrderFront(nil)
            return
        }
        NSApp.activate(ignoringOtherApps: true)
    }
}

// A menu item that runs a closure.
final class ActionItem: NSMenuItem {
    private let handler: () -> Void
    init(_ title: String, _ handler: @escaping () -> Void) {
        self.handler = handler
        super.init(title: title, action: nil, keyEquivalent: "")
        self.target = self
        self.action = #selector(fire)
    }
    required init(coder: NSCoder) { fatalError() }
    @objc private func fire() { handler() }
}

// MARK: - CLI one-shots

enum CLI {
    static func run(_ args: [String]) {
        let config = Config()
        let engine = Engine.shared
        func portArg() -> Int {
            if let i = args.firstIndex(of: "--port"), i + 1 < args.count, let v = Int(args[i + 1]) { return v }
            return config.settings.port
        }
        switch args.first ?? "" {
        case "--models":
            for m in engine.models(root: config.settings.splashRoot) {
                print("\(m.id)  \(String(format: "%.1f GB", m.sizeGB))")
            }
        case "--status":
            let port = portArg()
            guard let st = engine.statusSync(port) else {
                print("el engine no responde en :\(port)"); return
            }
            let v = StatusParser.parse(st)
            print("modelo       \(v.model)")
            print("decode       \(v.decodeTps.map { String(format: "%.1f tok/s", $0) } ?? "—")  ·  ITL \(v.itlMs.map(Fmt.ms) ?? "—")")
            print("prefill      \(v.prefillTps.map { String(format: "%.0f tok/s", $0) } ?? "—")  ·  TTFT \(v.ttftMs.map(Fmt.ms) ?? "—")")
            print("requests     \(v.submitted) (\(v.completed) ok, \(v.cancelled) cancel, \(v.failed) fallo)")
            print("kv           \(v.kvBytes.map(Fmt.bytes) ?? "—")  ·  ctx \(v.ctxUsed.map(Fmt.tok) ?? "—")/\(v.ctxMax.map(Fmt.tok) ?? "—")")
            print("memoria      \(v.memAlloc.map(Fmt.bytes) ?? "—")  ·  presión \(v.pressure ?? "—")")
            for f in v.inFlight { print("  \(InFlight.line(f))") }
        case "--stop":
            engine.adoptIfRunning()
            guard let pid = (engine.rawStatus?["instance"] as? [String: Any])?["pid"].flatMap({ Int32($0 as? Int ?? 0) }) else {
                print("no hay engine corriendo"); return
            }
            ProcTree.killTreeSync(pid)
            print("engine \(pid) detenido")
        case "--release":
            engine.releaseMemory(portArg())
            print("memoria solicitada para liberar")
        case "--logs":
            print(Config.logFileURL().path)
        default:
            print("usos: splash-studio --models | --status [--port N] | --stop | --release | --logs")
        }
    }
}
