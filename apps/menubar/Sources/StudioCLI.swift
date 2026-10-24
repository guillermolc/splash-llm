import Foundation

/// One-shot commands share the GUI's transport and validated configuration.
/// They never start a model or take ownership of an existing server.
@MainActor
enum StudioCLI {
    static func run(_ arguments: [String]) async -> Int32 {
        do {
            let command = arguments.first ?? "--help"
            var port = ConfigStore.shared.settings.port
            var root = ConfigStore.shared.settings.splashRoot
            var index = 1
            while index < arguments.count {
                let flag = arguments[index]
                guard index + 1 < arguments.count else { throw CommandError.invalid("Falta el valor de \(flag).") }
                let value = arguments[index + 1]
                switch flag {
                case "--port":
                    guard let parsed = Int(value), (1...65535).contains(parsed) else {
                        throw CommandError.invalid("El puerto debe estar entre 1 y 65535.")
                    }
                    port = parsed
                case "--root": root = value
                default: throw CommandError.invalid("Opción desconocida: \(flag).")
                }
                index += 2
            }
            let client = StatusClient()
            switch command {
            case "--models":
                for model in await Engine.shared.models(root: root) {
                    let size = model.sizeGB > 0 ? String(format: "%.1f GB", model.sizeGB) : "tamaño no disponible"
                    print("\(model.id)  \(size)")
                }
            case "--status":
                let status = try await client.status(port: port)
                print("modelo       \(status.model)")
                print("estado       \(status.ready ? "listo" : "iniciando")")
                print("generación   \(status.liveDecodeTps.map { String(format: "%.1f tok/s", $0) } ?? "—")")
                print("media        \(status.decodeTps.map { String(format: "%.1f tok/s", $0) } ?? "—")")
                print("memoria      \(status.memAlloc.map(Fmt.bytes) ?? "—")")
                print("KV residente \(status.kvBytes.map(Fmt.bytes) ?? "—")")
                print("SSD          \(status.ssdUsed.map(Fmt.bytes) ?? "—")")
                for request in status.inFlight { print("  \(InFlight.line(request))") }
            case "--release":
                try await client.releaseMemory(port: port)
                print("Pesos descargados de memoria; el servidor sigue disponible.")
            case "--stop":
                let status = try await client.status(port: port)
                guard let pid = status.pid else { throw CommandError.invalid("El servidor no identifica su proceso.") }
                try await ProcTree.terminateServer(pid: pid, port: port)
                print("Servidor de :\(port) detenido.")
            case "--logs": print(Config.logFileURL().path)
            case "--help":
                print("Splash Studio: --models [--root PATH] | --status [--port N] | --release [--port N] | --stop [--port N] | --logs")
            default: throw CommandError.invalid("Comando desconocido: \(command). Usá --help.")
            }
            return 0
        } catch {
            FileHandle.standardError.write(Data(("Splash Studio: \(error.localizedDescription)\n").utf8))
            return 1
        }
    }

    private enum CommandError: LocalizedError {
        case invalid(String)
        var errorDescription: String? {
            switch self { case .invalid(let message): return message }
        }
    }
}
