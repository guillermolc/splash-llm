import Foundation

struct Settings: Codable {
    var splashRoot: String
    var model: String
    var port: Int
    // Context: "auto" or a token count such as "32K".
    var context: String
    // "int8" or "bf16".
    var kvFormat: String
    // "off" or a duration such as "30m".
    var idleRelease: String
    var prefillInt8: Bool
    // "auto" or a size such as "28G".
    var maxMemory: String
    var useANE: Bool
    // "auto", "0" or a size such as "16G".
    var cacheDisk: String
    // Keep the SSD cache in ~/Library/Caches across restarts (warm starts).
    var persistentCache: Bool

    static let `default` = Settings(
        splashRoot: NSHomeDirectory() + "/Documents/Projects/splash-llm",
        model: "mlx-community/Qwen3.8-27B-4bit",
        port: 8001,
        context: "32K",
        kvFormat: "int8",
        idleRelease: "30m",
        prefillInt8: true,
        maxMemory: "auto",
        useANE: true,
        cacheDisk: "auto",
        persistentCache: true
    )
}

extension Settings {
    // Explicit so old config files (no "persistentCache" key) still decode.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        splashRoot = try c.decode(String.self, forKey: .splashRoot)
        model = try c.decode(String.self, forKey: .model)
        port = try c.decode(Int.self, forKey: .port)
        context = try c.decode(String.self, forKey: .context)
        kvFormat = try c.decode(String.self, forKey: .kvFormat)
        idleRelease = try c.decode(String.self, forKey: .idleRelease)
        prefillInt8 = try c.decode(Bool.self, forKey: .prefillInt8)
        maxMemory = try c.decode(String.self, forKey: .maxMemory)
        useANE = try c.decode(Bool.self, forKey: .useANE)
        cacheDisk = try c.decode(String.self, forKey: .cacheDisk)
        // Pre-flag configs default on, like Bionic/OMLX/MTPLX.
        persistentCache = try c.decodeIfPresent(Bool.self, forKey: .persistentCache) ?? true
    }
}

final class Config {
    private let fileURL: URL

    var settings: Settings

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let directory = home.appendingPathComponent("Library/Application Support/splash-menubar", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        fileURL = directory.appendingPathComponent("config.json")
        let logs = directory.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        if let data = try? Data(contentsOf: fileURL),
           let saved = try? JSONDecoder().decode(Settings.self, from: data) {
            settings = saved
        } else {
            settings = .default
        }
    }

    var engineLogURL: URL { Self.logFileURL() }

    /// Shared path so one-shots (CLI, old binaries) and the GUI agree.
    static func logFileURL() -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(
            "Library/Application Support/splash-menubar/logs/engine.log", isDirectory: false)
    }

    func save() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(settings).write(to: fileURL)
    }
}
