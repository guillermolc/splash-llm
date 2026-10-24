import AppKit
import Combine
import SwiftUI

@main
@MainActor
struct SplashStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var store = ConfigStore.shared
    @StateObject private var engine = Engine.shared

    var body: some Scene {
        Window("Splash Studio", id: "main") {
            StudioWindow(delegate: delegate)
                .environmentObject(store)
                .environmentObject(engine)
        }
        .defaultSize(width: 640, height: 520)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Mostrar Splash Studio") { delegate.openMainWindow() }
                    .keyboardShortcut("0", modifiers: .command)
            }
        }
        SwiftUI.Settings {
            StudioSettingsView()
                .environmentObject(store)
                .environmentObject(engine)
        }
    }
}

@MainActor
private struct StudioWindow: View {
    let delegate: AppDelegate
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        ContentView()
            .background(WindowVisibilityReader())
            .background {
                if #available(macOS 14, *) {
                    SettingsActionReader(delegate: delegate)
                }
            }
            .onAppear { delegate.showWindow = { openWindow(id: "main") } }
    }
}

@available(macOS 14, *)
@MainActor
private struct SettingsActionReader: View {
    let delegate: AppDelegate
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        Color.clear.frame(width: 0, height: 0)
            .onAppear { delegate.showSettings = { openSettings() } }
    }
}

/// Track the main window's visibility instead of application focus.
@MainActor
private struct WindowVisibilityReader: NSViewRepresentable {
    func makeNSView(context: Context) -> VisibilityView { VisibilityView() }
    func updateNSView(_ view: VisibilityView, context: Context) {}

    final class VisibilityView: NSView {
        private var observers: [NSObjectProtocol] = []

        override func viewDidMoveToWindow() {
            observers.forEach(NotificationCenter.default.removeObserver)
            observers = []
            guard let window else {
                Engine.shared.setVisible(false)
                return
            }
            for name in [NSWindow.didChangeOcclusionStateNotification,
                         NSWindow.didMiniaturizeNotification,
                         NSWindow.didDeminiaturizeNotification] {
                observers.append(NotificationCenter.default.addObserver(
                    forName: name, object: window, queue: .main
                ) { [weak self] _ in
                    MainActor.assumeIsolated { self?.updateVisibility() }
                })
            }
            updateVisibility()
        }

        private func updateVisibility() {
            Engine.shared.setVisible(window?.occlusionState.contains(.visible) == true)
        }

        deinit { observers.forEach(NotificationCenter.default.removeObserver) }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    var showWindow: (() -> Void)?
    var showSettings: (() -> Void)?
    private var statusItem: NSStatusItem?
    private var stateSubscription: AnyCancellable?
    private var terminating = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.first?.hasPrefix("--") == true {
            NSApp.setActivationPolicy(.prohibited)
            NSApp.windows.forEach { $0.orderOut(nil) }
            Task {
                let code = await StudioCLI.run(arguments)
                fflush(stdout)
                fflush(stderr)
                exit(code)
            }
            return
        }
        NSApp.setActivationPolicy(.regular)
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = NSImage(systemSymbolName: "water.waves", accessibilityDescription: "Splash Studio")
        item.button?.image?.isTemplate = true
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu
        statusItem = item
        stateSubscription = Engine.shared.$state.sink { [weak self] _ in
            Task { @MainActor in self?.updateMenuBar() }
        }
        Task { await Engine.shared.adoptIfRunning(port: ConfigStore.shared.settings.port) }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        openMainWindow()
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard Engine.shared.ownership == .owned else { return .terminateNow }
        if !terminating {
            terminating = true
            Task {
                await Engine.shared.stop()
                if Engine.shared.ownership == .owned {
                    terminating = false
                    openMainWindow()
                    sender.reply(toApplicationShouldTerminate: false)
                } else {
                    sender.reply(toApplicationShouldTerminate: true)
                }
            }
        }
        return .terminateLater
    }

    func openMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        if let showWindow { showWindow() }
        else { NSApp.windows.first(where: { $0.title == "Splash Studio" })?.makeKeyAndOrderFront(nil) }
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        let engine = Engine.shared
        menu.removeAllItems()
        let status = NSMenuItem(title: "Splash Studio · \(stateTitle(engine.state))", action: nil, keyEquivalent: "")
        status.isEnabled = false
        menu.addItem(status)
        if let snapshot = engine.snapshot {
            let model = NSMenuItem(title: snapshot.model, action: nil, keyEquivalent: "")
            model.isEnabled = false
            menu.addItem(model)
        }
        menu.addItem(.separator())
        menu.addItem(ActionItem("Mostrar ventana") { self.openMainWindow() })
        let chat = ActionItem("Abrir chat") { NSWorkspace.shared.open(engine.endpoint) }
        chat.isEnabled = engine.state == .running
        menu.addItem(chat)
        menu.addItem(ActionItem("Copiar URL de API") {
            let configuredPort = ConfigStore.shared.settings.port
            guard engine.ownership != .none || (1...65535).contains(configuredPort) else { return }
            NSPasteboard.general.clearContents()
            let endpoint = engine.ownership == .none
                ? URL(string: "http://127.0.0.1:\(configuredPort)")! : engine.endpoint
            NSPasteboard.general.setString(endpoint.appendingPathComponent("v1").absoluteString, forType: .string)
        })
        menu.addItem(.separator())
        switch engine.state {
        case .stopped:
            menu.addItem(ActionItem("Iniciar") {
                let settings = ConfigStore.shared.settings
                Task {
                    do { try await engine.start(root: settings.splashRoot, model: settings.model, s: settings) }
                    catch { self.showError(error) }
                }
            })
        case .failed:
            menu.addItem(ActionItem("Revisar error") { self.openMainWindow() })
            if engine.ownership != .none {
                menu.addItem(ActionItem(engine.ownership == .external ? "Desconectar" : "Detener") {
                    Task { await engine.stop() }
                })
            }
        case .running:
            menu.addItem(ActionItem(engine.ownership == .external ? "Desconectar" : "Detener") {
                Task { await engine.stop() }
            })
        case .starting, .stopping:
            let item = NSMenuItem(title: stateTitle(engine.state), action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        }
        menu.addItem(ActionItem("Ajustes…") {
            NSApp.activate(ignoringOtherApps: true)
            if let showSettings = self.showSettings { showSettings() }
            else { NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil) }
        })
        menu.addItem(.separator())
        menu.addItem(ActionItem("Salir") { NSApp.terminate(nil) })
    }

    private func updateMenuBar() {
        statusItem?.button?.toolTip = "Splash Studio · \(stateTitle(Engine.shared.state))"
    }

    private func showError(_ error: Error) {
        openMainWindow()
        let alert = NSAlert()
        alert.messageText = "No se pudo iniciar Splash"
        alert.informativeText = error.localizedDescription
        if let window = NSApp.keyWindow { alert.beginSheetModal(for: window) }
        else { alert.runModal() }
    }

    private func stateTitle(_ state: EngineState) -> String {
        switch state {
        case .stopped: return "Detenido"
        case .starting: return "Iniciando"
        case .running: return "Listo"
        case .stopping: return "Deteniendo"
        case .failed: return "Error"
        }
    }
}

@MainActor
final class ActionItem: NSMenuItem {
    private let handler: () -> Void
    init(_ title: String, _ handler: @escaping () -> Void) {
        self.handler = handler
        super.init(title: title, action: nil, keyEquivalent: "")
        target = self
        action = #selector(fire)
    }
    required init(coder: NSCoder) { fatalError("ActionItem is created programmatically") }
    @objc private func fire() { handler() }
}
